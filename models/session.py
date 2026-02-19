from __future__ import annotations

import json
import os
import uuid
import time
import pandas as pd
import logging
from config.config import BASE_DATA_DIR

logger = logging.getLogger(__name__)

META_FILENAME = "meta.json"

# Upload file naming conventions used by the UI/backend.
UPLOAD_FILENAMES = {
    "identities": "identities.csv",
    "assignments": "assignments.csv",
    "entitlements": "entitlements.csv",
}


def _write_meta(session_path: str, owner: dict) -> None:
    meta = {
        "created": int(time.time()),
        "owner": {
            "sub": owner.get("sub"),
            "email": owner.get("email"),
            "name": owner.get("name"),
            "hd": owner.get("hd"),
        },
    }
    with open(os.path.join(session_path, META_FILENAME), "w") as f:
        json.dump(meta, f, indent=2)


def _read_meta(session_path: str) -> dict:
    path = os.path.join(session_path, META_FILENAME)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def create_session(owner: dict) -> str:
    session_id = str(uuid.uuid4())
    session_path = os.path.join(BASE_DATA_DIR, session_id)
    os.makedirs(os.path.join(session_path, "uploads"), exist_ok=True)
    os.makedirs(os.path.join(session_path, "processed"), exist_ok=True)
    os.makedirs(os.path.join(session_path, "results"), exist_ok=True)

    _write_meta(session_path, owner)
    return session_id


def list_sessions(owner_sub: str | None = None) -> list[dict]:
    """List sessions. If owner_sub is provided, only return sessions owned by that user."""
    os.makedirs(BASE_DATA_DIR, exist_ok=True)
    sessions = []

    for name in os.listdir(BASE_DATA_DIR):
        path = os.path.join(BASE_DATA_DIR, name)
        if not os.path.isdir(path):
            continue

        meta = _read_meta(path)
        meta_owner_sub = (meta.get("owner") or {}).get("sub")
        if owner_sub and meta_owner_sub != owner_sub:
            continue

        uploads_dir = os.path.join(path, "uploads")
        has_uploads = (
            os.path.isdir(uploads_dir)
            and any(f.endswith(".csv") for f in os.listdir(uploads_dir))
        )

        has_results = os.path.isfile(os.path.join(path, "results", "results.json"))
        mtime = os.path.getmtime(path)

        stats = {}
        stats_path = os.path.join(path, "processed", "stats.json")
        if os.path.isfile(stats_path):
            try:
                with open(stats_path) as f:
                    stats = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        sessions.append(
            {
                "session_id": name,
                "modified": mtime,
                "has_uploads": has_uploads,
                "has_results": has_results,
                "total_users": stats.get("total_users"),
                "total_entitlements": stats.get("total_entitlements"),
                "total_assignments": stats.get("total_assignments"),
                "apps": stats.get("apps"),
                "owner": meta.get("owner"),
                "created": meta.get("created"),
            }
        )

    sessions.sort(key=lambda s: s["modified"], reverse=True)
    return sessions


def get_session_path(session_id: str) -> str:
    """Return the absolute on-disk path for a session.

    Security: validate the session_id and prevent path traversal outside BASE_DATA_DIR.
    """
    # Ensure the path component is a UUID (all sessions are created with UUIDs).
    try:
        uuid.UUID(session_id)
    except Exception as e:
        raise ValueError("Invalid session_id") from e

    base_dir_abs = os.path.abspath(BASE_DATA_DIR)
    session_path = os.path.abspath(os.path.join(base_dir_abs, session_id))

    # Prevent traversal (e.g., session_id=../../etc)
    if not (session_path == base_dir_abs or session_path.startswith(base_dir_abs + os.sep)):
        raise ValueError("Invalid session_id")

    logger.info(f"Session path: {session_path}")
    if not os.path.isdir(session_path):
        raise FileNotFoundError(f"Session {session_id} not found")
    return session_path


def session_owner_sub(session_id: str) -> str | None:
    session_path = get_session_path(session_id)
    meta = _read_meta(session_path)
    return (meta.get("owner") or {}).get("sub")


def save_json(session_id: str, filename: str, data: dict) -> None:
    session_path = get_session_path(session_id)
    filepath = os.path.join(session_path, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_json(session_id: str, filename: str) -> dict:
    session_path = get_session_path(session_id)
    filepath = os.path.join(session_path, filename)
    with open(filepath, "r") as f:
        return json.load(f)


def save_dataframe(session_id: str, filename: str, df: pd.DataFrame) -> None:
    session_path = get_session_path(session_id)
    filepath = os.path.join(session_path, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=True)


def load_dataframe(session_id: str, filename: str) -> pd.DataFrame:
    session_path = get_session_path(session_id)
    filepath = os.path.join(session_path, filename)
    if "matrix.csv" in filename:
        return pd.read_csv(filepath, index_col=0)
    return pd.read_csv(filepath)


def _count_csv_data_rows(filepath: str) -> int:
    """Return the number of data rows in a CSV (excluding the header row).

    This is intentionally implemented without pandas for performance on large
    files (e.g., millions of lines). It assumes the file has a single header row.
    """
    # Fast path: count newlines in binary chunks.
    # This counts physical lines; it will not handle embedded newlines inside
    # quoted CSV fields. For our role-mining CSVs, that is an acceptable constraint.
    line_count = 0
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)  # 1MB
            if not chunk:
                break
            line_count += chunk.count(b"\n")

    # If file ends without a trailing newline, count the last line.
    try:
        with open(filepath, "rb") as f:
            f.seek(-1, os.SEEK_END)
            last = f.read(1)
            if last and last != b"\n":
                line_count += 1
    except OSError:
        # Empty file
        return 0

    # Subtract header row if present.
    return max(0, line_count - 1)


def get_upload_file_info(session_id: str, file_type: str) -> dict | None:
    """Compute upload file metadata on demand.

    Returns None if the file is not present.

    Output shape (when present):
      {
        "present": True,
        "filename": "identities.csv",
        "size_bytes": 1234,
        "row_count": 5678
      }
    """
    filename = UPLOAD_FILENAMES.get(file_type)
    if not filename:
        raise ValueError(f"Unknown file_type: {file_type}")

    session_path = get_session_path(session_id)
    uploads_dir = os.path.join(session_path, "uploads")
    filepath = os.path.join(uploads_dir, filename)
    if not os.path.isfile(filepath):
        return None

    try:
        size_bytes = os.path.getsize(filepath)
    except OSError:
        size_bytes = None

    row_count = None
    try:
        row_count = _count_csv_data_rows(filepath)
    except Exception as e:
        logger.warning("Failed to count rows for %s (%s): %s", filepath, file_type, e)

    return {
        "present": True,
        "filename": filename,
        "size_bytes": size_bytes,
        "row_count": row_count,
    }


def get_uploaded_files_info(session_id: str) -> dict:
    """Return per-upload metadata for the known upload CSVs."""
    return {
        "identities": get_upload_file_info(session_id, "identities"),
        "assignments": get_upload_file_info(session_id, "assignments"),
        "entitlements": get_upload_file_info(session_id, "entitlements"),
    }
