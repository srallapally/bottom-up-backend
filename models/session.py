import json
import os
import uuid

import pandas as pd
import logging


from config.config import BASE_DATA_DIR
logger = logging.getLogger(__name__)

def create_session() -> str:
    session_id = str(uuid.uuid4())
    session_path = os.path.join(BASE_DATA_DIR, session_id)
    os.makedirs(os.path.join(session_path, "uploads"), exist_ok=True)
    os.makedirs(os.path.join(session_path, "processed"), exist_ok=True)
    os.makedirs(os.path.join(session_path, "results"), exist_ok=True)
    return session_id


def list_sessions() -> list[dict]:
    """List all sessions with metadata (created time, status)."""
    os.makedirs(BASE_DATA_DIR, exist_ok=True)
    sessions = []
    for name in os.listdir(BASE_DATA_DIR):
        path = os.path.join(BASE_DATA_DIR, name)
        if not os.path.isdir(path):
            continue

        has_uploads = any(
            f.endswith(".csv")
            for f in os.listdir(os.path.join(path, "uploads"))
        ) if os.path.isdir(os.path.join(path, "uploads")) else False

        has_results = os.path.isfile(os.path.join(path, "results", "results.json"))

        # Use folder mtime as last modified
        mtime = os.path.getmtime(path)

        # Try to load stats for user/entitlement counts
        stats = {}
        stats_path = os.path.join(path, "processed", "stats.json")
        if os.path.isfile(stats_path):
            try:
                with open(stats_path) as f:
                    stats = json.load(f)
            except (json.JSONDecodeError, IOError):
                pass

        sessions.append({
            "session_id": name,
            "modified": mtime,
            "has_uploads": has_uploads,
            "has_results": has_results,
            "total_users": stats.get("total_users"),
            "total_entitlements": stats.get("total_entitlements"),
            "apps": stats.get("apps"),
        })

    sessions.sort(key=lambda s: s["modified"], reverse=True)
    return sessions


def get_session_path(session_id: str) -> str:
    session_path = os.path.join(BASE_DATA_DIR, session_id)
    logger.info(f"Session path: {session_path}")
    if not os.path.isdir(session_path):
        raise FileNotFoundError(f"Session {session_id} not found")
    return session_path


def save_json(session_id: str, filename: str, data: dict) -> None:
    session_path = get_session_path(session_id)
    filepath = os.path.join(session_path, filename)
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
    df.to_csv(filepath, index=True)


def load_dataframe(session_id: str, filename: str) -> pd.DataFrame:
    session_path = get_session_path(session_id)
    filepath = os.path.join(session_path, filename)
    if 'matrix.csv' in filename:
        return pd.read_csv(filepath, index_col=0)
    else:
        return pd.read_csv(filepath)
    # return pd.read_csv(filepath, index_col=0)