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
    session_path = os.path.join(BASE_DATA_DIR, session_id)
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
