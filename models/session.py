from __future__ import annotations

import json
import logging
import os
import time
import uuid
from datetime import datetime, timezone

import pandas as pd

from config.config import BASE_DATA_DIR

logger = logging.getLogger(__name__)

META_FILENAME = "meta.json"

# Upload file naming conventions used by the UI/backend.
UPLOAD_FILENAMES = {
    "identities": "identities.csv",
    "assignments": "assignments.csv",
    "entitlements": "entitlements.csv",
}


# =============================================================================
# Storage backend selection
# =============================================================================
# Dev (local): filesystem under BASE_DATA_DIR
# GCP: Firestore (session registry/ownership/status/listing) + GCS (session artifacts)
_SESSION_BACKEND = os.getenv("SESSION_BACKEND", "local").strip().lower()
_GCS_BUCKET = os.getenv("GCS_BUCKET")
_FIRESTORE_COLLECTION = os.getenv("FIRESTORE_SESSIONS_COLLECTION", "sessions")

# If true, refresh the /tmp session cache on each get_session_path().
# Default false to avoid repeated downloads.
_REFRESH_CACHE = os.getenv("SESSION_CACHE_REFRESH", "false").strip().lower() in {"1", "true", "yes"}


def _use_gcp_backend() -> bool:
    return _SESSION_BACKEND in {"gcp", "cloud", "firestore", "gcs"}


def _require_gcp_config() -> None:
    if not _GCS_BUCKET:
        raise RuntimeError("GCS_BUCKET is required when SESSION_BACKEND is gcp")


def _gcp_clients():
    """Lazy-import GCP clients so local dev doesn't require them."""
    from google.cloud import firestore  # type: ignore
    from google.cloud import storage  # type: ignore

    return storage.Client(), firestore.Client()


def _gcp_session_doc(session_id: str) -> dict | None:
    _require_gcp_config()
    _, fs = _gcp_clients()
    doc = fs.collection(_FIRESTORE_COLLECTION).document(session_id).get()
    if not doc.exists:
        return None
    return doc.to_dict() or {}


def _gcp_session_prefix(session_id: str) -> str:
    """Return the GCS prefix for a session.

    Stored in Firestore at sessions/{sessionId}.gcsPrefix.
    """
    doc = _gcp_session_doc(session_id)
    if not doc:
        raise FileNotFoundError(f"Session {session_id} not found")
    prefix = (doc.get("gcsPrefix") or "").lstrip("/")
    if not prefix:
        raise RuntimeError(f"Session {session_id} is missing gcsPrefix")
    if not prefix.endswith("/"):
        prefix += "/"
    return prefix


def _gcp_local_cache_dir(session_id: str) -> str:
    # Cloud Run writable space is /tmp
    return os.path.join("/tmp", "bottom-up-sessions", session_id)


def _gcp_download_prefix(prefix: str, local_dir: str) -> None:
    _require_gcp_config()
    st, _ = _gcp_clients()
    bucket = st.bucket(_GCS_BUCKET)

    os.makedirs(local_dir, exist_ok=True)

    for blob in st.list_blobs(bucket, prefix=prefix):
        rel = blob.name[len(prefix) :]
        if not rel or rel.endswith("/"):
            continue
        dest = os.path.join(local_dir, rel)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        blob.download_to_filename(dest)


def _gcp_upload_file(prefix: str, local_dir: str, rel_path: str) -> None:
    _require_gcp_config()
    st, _ = _gcp_clients()
    bucket = st.bucket(_GCS_BUCKET)

    local_path = os.path.join(local_dir, rel_path)
    if not os.path.isfile(local_path):
        raise FileNotFoundError(local_path)

    blob_name = prefix + rel_path.replace(os.sep, "/")
    bucket.blob(blob_name).upload_from_filename(local_path)


def sync_file(session_id: str, rel_path: str) -> None:
    """Sync a single file to the configured backend.

    Local backend: no-op.
    GCP backend: upload the file at rel_path from the local session dir to GCS.
    """
    if not _use_gcp_backend():
        return

    prefix = _gcp_session_prefix(session_id)
    local_dir = _gcp_local_cache_dir(session_id)
    # FIX 21: log each GCS file upload with size for observability.
    local_path = os.path.join(local_dir, rel_path)
    size_bytes = 0
    try:
        size_bytes = os.path.getsize(local_path)
    except OSError:
        pass
    logger.info("gcs_sync_file session=%s path=%s bytes=%d", session_id, rel_path, size_bytes)
    _gcp_upload_file(prefix, local_dir, rel_path)


def sync_session_tree(session_id: str) -> None:
    """Upload all files under the local session directory to GCS.

    Intended for Cloud Run Jobs after processing/mining.
    """
    if not _use_gcp_backend():
        return

    prefix = _gcp_session_prefix(session_id)
    local_dir = _gcp_local_cache_dir(session_id)

    # FIX 21: track file count, total bytes, elapsed for GCS sync observability.
    _t = time.time()
    _file_count = 0
    _total_bytes = 0
    for root, _, files in os.walk(local_dir):
        for fname in files:
            local_path = os.path.join(root, fname)
            rel = os.path.relpath(local_path, local_dir).replace(os.sep, "/")
            try:
                _total_bytes += os.path.getsize(local_path)
            except OSError:
                pass
            _gcp_upload_file(prefix, local_dir, rel)
            _file_count += 1
    logger.info(
        "gcs_sync_tree_complete session=%s files=%d bytes=%d elapsed_ms=%.0f",
        session_id, _file_count, _total_bytes, (time.time() - _t) * 1000,
    )


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _read_json(path: str) -> dict:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def create_session(owner: dict) -> str:
    """Create a new role-mining session."""
    session_id = str(uuid.uuid4())

    if not _use_gcp_backend():
        session_path = os.path.join(BASE_DATA_DIR, session_id)
        os.makedirs(os.path.join(session_path, "uploads"), exist_ok=True)
        os.makedirs(os.path.join(session_path, "processed"), exist_ok=True)
        os.makedirs(os.path.join(session_path, "results"), exist_ok=True)

        meta = {
            "created": int(time.time()),
            "updated": int(time.time()),
            "status": "CREATED",
            "owner": {
                "sub": owner.get("sub"),
                "email": owner.get("email"),
                "name": owner.get("name"),
                "hd": owner.get("hd"),
            },
        }
        _write_json(os.path.join(session_path, META_FILENAME), meta)
        return session_id

    _require_gcp_config()
    st, fs = _gcp_clients()

    owner_sub = owner.get("sub")
    if not owner_sub:
        raise ValueError("owner.sub is required")

    prefix = f"users/{owner_sub}/sessions/{session_id}/"

    doc = {
        "id": session_id,
        "ownerSub": owner_sub,
        "owner": {
            "sub": owner_sub,
            "email": owner.get("email"),
            "name": owner.get("name"),
            "hd": owner.get("hd"),
        },
        "createdAt": _iso_now(),
        "updatedAt": _iso_now(),
        "status": "CREATED",
        "gcsBucket": _GCS_BUCKET,
        "gcsPrefix": prefix,
    }
    fs.collection(_FIRESTORE_COLLECTION).document(session_id).set(doc)

    # Initialize local cache and meta.json so existing code paths work.
    local_dir = _gcp_local_cache_dir(session_id)
    os.makedirs(os.path.join(local_dir, "uploads"), exist_ok=True)
    os.makedirs(os.path.join(local_dir, "processed"), exist_ok=True)
    os.makedirs(os.path.join(local_dir, "results"), exist_ok=True)

    meta = {
        "created": int(time.time()),
        "updated": int(time.time()),
        "status": "CREATED",
        "owner": doc["owner"],
    }
    _write_json(os.path.join(local_dir, META_FILENAME), meta)

    bucket = st.bucket(_GCS_BUCKET)
    bucket.blob(prefix + META_FILENAME).upload_from_string(
        json.dumps(meta, indent=2), content_type="application/json"
    )

    return session_id


def list_sessions(owner_sub: str | None = None) -> list[dict]:
    """List sessions.

    The UI expects a list of dicts (at minimum containing session_id or id).
    """
    if not _use_gcp_backend():
        os.makedirs(BASE_DATA_DIR, exist_ok=True)
        sessions: list[dict] = []
        for name in os.listdir(BASE_DATA_DIR):
            path = os.path.join(BASE_DATA_DIR, name)
            if not os.path.isdir(path):
                continue

            meta = _read_json(os.path.join(path, META_FILENAME))
            if owner_sub and meta.get("owner", {}).get("sub") != owner_sub:
                continue

            sessions.append(
                {
                    "session_id": name,
                    "modified": os.path.getmtime(path),
                    "created": meta.get("created"),
                    "status": meta.get("status"),
                }
            )

        sessions.sort(key=lambda s: s.get("modified") or 0, reverse=True)
        return sessions

    _require_gcp_config()
    _, fs = _gcp_clients()

    q = fs.collection(_FIRESTORE_COLLECTION)
    if owner_sub:
        q = q.where("ownerSub", "==", owner_sub)

    # Best-effort ordering. If index is missing, Firestore raises; UI can still render.
    try:
        q = q.order_by("updatedAt", direction="DESCENDING")
    except Exception:
        pass

    out: list[dict] = []
    for doc in q.stream():
        d = doc.to_dict() or {}
        out.append(
            {
                "id": d.get("id") or doc.id,
                "session_id": d.get("id") or doc.id,
                "created": d.get("createdAt"),
                "updated": d.get("updatedAt"),
                "status": d.get("status"),
                "name": d.get("name"),
                "summary": d.get("summary"),
            }
        )

    return out


def get_session_path(session_id: str) -> str:
    """Return a local directory for the session.

    Local: BASE_DATA_DIR/<session_id>
    GCP: /tmp/bottom-up-sessions/<session_id> (downloaded from GCS on first use)
    """
    if not _use_gcp_backend():
        session_path = os.path.join(BASE_DATA_DIR, session_id)
        logger.info(f"Session path: {session_path}")
        if not os.path.isdir(session_path):
            raise FileNotFoundError(f"Session {session_id} not found")
        return session_path

    prefix = _gcp_session_prefix(session_id)
    local_dir = _gcp_local_cache_dir(session_id)

    if _REFRESH_CACHE or not os.path.isdir(local_dir):
        os.makedirs(local_dir, exist_ok=True)
        # FIX 15: Skip download when the session has no meaningful artifacts yet.
        # A freshly created session contains only meta.json in GCS; downloading
        # it wastes a GCS list + download call and overwrites the copy that
        # create_session() already wrote locally.
        # Count non-metadata blobs; only download if any exist.
        _st, _ = _gcp_clients()
        _bucket = _st.bucket(_GCS_BUCKET)
        _has_artifacts = any(
            b.name != prefix + META_FILENAME
            for b in _st.list_blobs(_bucket, prefix=prefix, max_results=2)
        )
        if _has_artifacts:
            _gcp_download_prefix(prefix, local_dir)

    # Ensure expected subdirs exist even if prefix is partially populated.
    os.makedirs(os.path.join(local_dir, "uploads"), exist_ok=True)
    os.makedirs(os.path.join(local_dir, "processed"), exist_ok=True)
    os.makedirs(os.path.join(local_dir, "results"), exist_ok=True)
    return local_dir


def session_owner_sub(session_id: str) -> str | None:
    if not _use_gcp_backend():
        try:
            session_path = get_session_path(session_id)
        except FileNotFoundError:
            return None
        meta = _read_json(os.path.join(session_path, META_FILENAME))
        return meta.get("owner", {}).get("sub")

    doc = _gcp_session_doc(session_id)
    if not doc:
        return None

    # Backward-compatible reads for older Firestore schemas.
    # New schema: ownerSub (top-level) and owner.sub (nested).
    # Legacy candidates: owner_sub (snake_case) and nested variants.
    owner = doc.get("owner") or {}
    return (
        doc.get("ownerSub")
        or doc.get("owner_sub")
        or owner.get("sub")
        or owner.get("ownerSub")
        or owner.get("owner_sub")
    )


def update_session_fields(session_id: str, fields: dict) -> None:
    """Best-effort session metadata update.

    Local: updates meta.json.
    GCP: patches Firestore, and also updates meta.json in the local cache + GCS.
    """
    # FIX 22: log status transitions at the persistence layer so all callers
    # (Flask routes, worker, tests) produce a consistent audit trail without
    # each needing to implement transition logging themselves.
    new_status = fields.get("status")
    if new_status:
        logger.info("session_status_set session=%s status=%s", session_id, new_status)

    if not _use_gcp_backend():
        session_path = get_session_path(session_id)
        meta_path = os.path.join(session_path, META_FILENAME)
        meta = _read_json(meta_path)
        old_status = meta.get("status")
        meta.update(fields)
        meta["updated"] = int(time.time())
        _write_json(meta_path, meta)
        if new_status and old_status != new_status:
            logger.info(
                "session_status_transition session=%s %s -> %s",
                session_id, old_status or "?", new_status,
            )
        return

    _require_gcp_config()
    st, fs = _gcp_clients()

    patch = dict(fields)
    patch["updatedAt"] = _iso_now()
    fs.collection(_FIRESTORE_COLLECTION).document(session_id).set(patch, merge=True)

    # Keep meta.json consistent for any code paths that still read it.
    local_dir = _gcp_local_cache_dir(session_id)
    os.makedirs(local_dir, exist_ok=True)
    meta_path = os.path.join(local_dir, META_FILENAME)
    meta = _read_json(meta_path)
    meta.update(fields)
    meta["updated"] = int(time.time())
    _write_json(meta_path, meta)

    prefix = _gcp_session_prefix(session_id)
    bucket = st.bucket(_GCS_BUCKET)
    bucket.blob(prefix + META_FILENAME).upload_from_string(
        json.dumps(meta, indent=2), content_type="application/json"
    )


def save_json(session_id: str, filename: str, data: dict) -> None:
    session_path = get_session_path(session_id)
    filepath = os.path.join(session_path, filename)
    _write_json(filepath, data)
    sync_file(session_id, filename)


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
    sync_file(session_id, filename)


def load_dataframe(session_id: str, filename: str) -> pd.DataFrame:
    session_path = get_session_path(session_id)
    filepath = os.path.join(session_path, filename)
    if "matrix.csv" in filename:
        return pd.read_csv(filepath, index_col=0)
    return pd.read_csv(filepath)