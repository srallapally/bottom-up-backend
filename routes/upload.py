import json
import os
from datetime import datetime, timezone
from flask import Blueprint, request, jsonify, g

from models.session import (
    create_session,
    get_session_path,
    sync_file,
    save_json,
    list_sessions,
    session_owner_sub,
    update_session_fields,
    sync_session_tree,
)
from services.data_loader import process_upload
from services.auth import require_auth
from services.cloud_run_jobs import run_process_job

upload_bp = Blueprint("upload", __name__)

VALID_FILE_TYPES = {"identities", "assignments", "entitlements"}

# Guardrail: reject role-mining CSVs that are too large to process safely.
# Configure via env for GCP deployments.
# Example: MAX_CSV_ROWS=2000000
try:
    MAX_CSV_ROWS = int(os.getenv("MAX_CSV_ROWS", "2000000"))
except ValueError:
    MAX_CSV_ROWS = 2000000


def _count_csv_data_rows(filepath: str) -> int:
    """Count CSV data rows (excluding the header row).

    Note: This counts physical lines and assumes no embedded newlines inside
    quoted CSV fields. For our role-mining CSVs this is acceptable and is
    significantly faster than pandas for multi-million line files.
    """
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


def _upload_file_info(uploads_dir: str, file_type: str) -> dict | None:
    """Return file metadata for an uploaded CSV, or None if not present."""
    path = os.path.join(uploads_dir, f"{file_type}.csv")
    if not os.path.isfile(path):
        return None

    size_bytes = None
    try:
        size_bytes = os.path.getsize(path)
    except OSError:
        pass

    row_count = None
    try:
        row_count = _count_csv_data_rows(path)
    except Exception:
        # Best-effort; keep None if counting fails
        pass

    return {
        "present": True,
        "filename": f"{file_type}.csv",
        "size_bytes": size_bytes,
        "row_count": row_count,
    }


def _ensure_owner(session_id: str):
    owner = session_owner_sub(session_id)
    if not owner:
        return jsonify({"error": "Session missing owner metadata"}), 403
    if owner != g.user["sub"]:
        return jsonify({"error": "Forbidden"}), 403
    return None


def _iso_from_epoch_seconds(epoch_seconds: float | int | None) -> str | None:
    if epoch_seconds is None:
        return None
    try:
        return datetime.fromtimestamp(float(epoch_seconds), tz=timezone.utc).isoformat().replace("+00:00", "Z")
    except Exception:
        return None


def _read_json_file(path: str) -> dict | None:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _derive_status(progress: dict) -> str:
    """
    Deterministic status derived from what exists on disk.
    This supports a sessions list UI without N+1 calls.
    """
    uploaded = progress["uploaded"]
    processed = progress["processed"]
    results_ready = progress["resultsReady"]

    if results_ready:
        return "ready"
    if processed:
        return "ready_to_mine"
    if uploaded["identities"] and uploaded["assignments"] and uploaded["entitlements"]:
        return "ready_to_process"
    if uploaded["identities"] or uploaded["assignments"] or uploaded["entitlements"]:
        return "uploading"
    return "created"


# -------------------------
# List sessions for user
# -------------------------
@upload_bp.route("/api/sessions", methods=["GET"])
@require_auth
def get_sessions():
    """
    UI-efficient sessions list:
      - includes status/progress/summary so the dashboard can render in 1 call
      - does NOT include heavy results payloads
    """
    # list_sessions is already owner-filtered in your implementation
    raw_sessions = list_sessions(owner_sub=g.user["sub"])

    sessions = []

    session_backend = os.getenv("SESSION_BACKEND", "local").strip().lower()
    if session_backend in {"gcp", "cloud", "firestore", "gcs"}:
        # In GCP mode, avoid downloading each session from GCS just to compute progress.
        for s in raw_sessions:
            session_id = s.get("session_id") or s.get("id")
            if not session_id:
                continue

            status = (s.get("status") or "created").lower()
            progress = {
                "uploaded": {"identities": False, "assignments": False, "entitlements": False},
                "processed": status in {"processed", "ready_to_mine", "mining", "mined", "ready"},
                "mined": status in {"mined", "ready"},
                "resultsReady": status in {"mined", "ready"},
            }

            sessions.append({
                "id": session_id,
                "name": s.get("name") or session_id,
                "status": status,
                "createdAt": s.get("created"),
                "updatedAt": s.get("updated"),
                "summary": s.get("summary"),
                "progress": progress,
                "links": {
                    "self": f"/api/sessions/{session_id}",
                    "status": f"/api/sessions/{session_id}/status",
                    "config": f"/api/sessions/{session_id}/config",
                    "results": f"/api/sessions/{session_id}/results",
                    "process": f"/api/sessions/{session_id}/process",
                    "mine": f"/api/sessions/{session_id}/mine",
                    "export": f"/api/sessions/{session_id}/export",
                },
            })

        return jsonify({"sessions": sessions}), 200

    for s in raw_sessions:
        session_id = s.get("session_id") or s.get("id")  # tolerate either key
        if not session_id:
            continue

        # Best-effort path resolution
        try:
            session_path = get_session_path(session_id)
        except FileNotFoundError:
            continue

        uploads_dir = os.path.join(session_path, "uploads")
        processed_dir = os.path.join(session_path, "processed")
        results_dir = os.path.join(session_path, "results")

        uploaded = {
            "identities": os.path.isfile(os.path.join(uploads_dir, "identities.csv")),
            "assignments": os.path.isfile(os.path.join(uploads_dir, "assignments.csv")),
            "entitlements": os.path.isfile(os.path.join(uploads_dir, "entitlements.csv")),
        }

        stats_path = os.path.join(processed_dir, "stats.json")
        stats = _read_json_file(stats_path)
        processed = stats is not None

        results_ready = (
            os.path.isfile(os.path.join(results_dir, "draft_results.json"))
            or os.path.isfile(os.path.join(results_dir, "results.json"))
        )

        progress = {
            "uploaded": uploaded,
            "processed": processed,
            "mined": results_ready,
            "resultsReady": results_ready,
        }

        # Use filesystem mtime as updatedAt (stable, cheap)
        updated_at = _iso_from_epoch_seconds(s.get("modified")) or _iso_from_epoch_seconds(os.path.getmtime(session_path))
        created_at = _iso_from_epoch_seconds(s.get("created"))  # if available from meta.json; otherwise omit

        # Optional: pick up a human-friendly name if meta.json exists
        meta = _read_json_file(os.path.join(session_path, "meta.json")) or {}
        name = meta.get("name") or s.get("name") or session_id

        status = _derive_status(progress)

        summary = None
        if stats:
            # tolerate either naming style in stats.json
            summary = {
                "totalUsers": stats.get("total_users") or stats.get("totalUsers") or stats.get("users"),
                "totalEntitlements": stats.get("total_entitlements") or stats.get("totalEntitlements") or stats.get("entitlements"),
                "totalAssignments": stats.get("total_assignments") or stats.get("totalAssignments") or stats.get("assignments"),
                "apps": stats.get("apps"),
            }

        sessions.append({
            "id": session_id,
            "name": name,
            "status": status,
            "createdAt": created_at,
            "updatedAt": updated_at,
            "summary": summary,
            "progress": progress,
            "links": {
                "self": f"/api/sessions/{session_id}",
                "status": f"/api/sessions/{session_id}/status",
                "config": f"/api/sessions/{session_id}/config",
                "results": f"/api/sessions/{session_id}/results",
            }
        })

    # Sort newest first by updatedAt (string ISO-Z sorts lexicographically)
    sessions.sort(key=lambda x: x.get("updatedAt") or "", reverse=True)

    default_session_id = sessions[0]["id"] if sessions else None

    return jsonify({
        "sessions": sessions,
        "defaultSessionId": default_session_id
    }), 200


# -------------------------
# Create session
# -------------------------
@upload_bp.route("/api/sessions", methods=["POST"])
@require_auth
def new_session():
    session_id = create_session(owner=g.user)
    return jsonify({"session_id": session_id}), 201


# -------------------------
# Upload file
# -------------------------
@upload_bp.route("/api/sessions/<session_id>/upload", methods=["POST"])
@require_auth
def upload_file(session_id):
    deny = _ensure_owner(session_id)
    if deny:
        return deny

    try:
        session_path = get_session_path(session_id)
    except FileNotFoundError:
        return jsonify({"error": "Session not found"}), 404

    file_type = request.form.get("file_type", "").strip().lower()
    if file_type not in VALID_FILE_TYPES:
        return jsonify({"error": f"file_type must be one of: {', '.join(sorted(VALID_FILE_TYPES))}"}), 400

    if "file" not in request.files:
        return jsonify({"error": "No file in request"}), 400

    f = request.files["file"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    dest = os.path.join(session_path, "uploads", f"{file_type}.csv")
    f.save(dest)

    # If running in GCP backend mode, persist the upload to GCS.
    sync_file(session_id, f"uploads/{file_type}.csv")

    # Guardrail: enforce a max rows limit before pandas-heavy processing.
    try:
        row_count = _count_csv_data_rows(dest)
        if row_count > MAX_CSV_ROWS:
            try:
                os.remove(dest)
            except OSError:
                pass
            return jsonify({
                "error": "CSV exceeds maximum allowed rows",
                "max_rows": MAX_CSV_ROWS,
                "row_count": row_count,
                "file_type": file_type,
            }), 400
    except Exception:
        # Best-effort: if counting fails, continue and let downstream validation handle it.
        pass

    return jsonify({
        "session_id": session_id,
        "file_type": file_type,
        "status": "uploaded"
    }), 200


# -------------------------
# Process uploaded files
# -------------------------
@upload_bp.route("/api/sessions/<session_id>/process", methods=["POST"])
@require_auth
def process_files(session_id):
    deny = _ensure_owner(session_id)
    if deny:
        return deny

    execution_backend = os.getenv("EXECUTION_BACKEND", "local").strip().lower()
    if execution_backend in {"gcp", "cloud", "cloud_run", "cloud_run_jobs", "jobs"}:
        # Trigger async processing via Cloud Run Jobs. Endpoint contract unchanged.
        try:
            update_session_fields(session_id, {"status": "PROCESSING"})
            op = run_process_job(session_id=session_id, owner_sub=g.user["sub"])
            return jsonify({
                "session_id": session_id,
                "status": "PROCESSING",
                "operation": op.get("name"),
            }), 202
        except FileNotFoundError:
            return jsonify({"error": "Session not found"}), 404
        except Exception as e:
            # Best-effort status update
            try:
                update_session_fields(session_id, {"status": "FAILED", "error": str(e)})
            except Exception:
                pass
            return jsonify({"error": str(e)}), 500
    try:
        session_path = get_session_path(session_id)
    except FileNotFoundError:
        return jsonify({"error": "Session not found"}), 404

    # Guardrail: re-check uploaded CSV sizes before starting expensive processing.
    uploads_dir = os.path.join(session_path, "uploads")
    for ft in ("identities", "assignments", "entitlements"):
        path = os.path.join(uploads_dir, f"{ft}.csv")
        if not os.path.isfile(path):
            continue
        try:
            row_count = _count_csv_data_rows(path)
            if row_count > MAX_CSV_ROWS:
                return jsonify({
                    "error": "CSV exceeds maximum allowed rows",
                    "max_rows": MAX_CSV_ROWS,
                    "row_count": row_count,
                    "file_type": ft,
                }), 400
        except Exception:
            # If counting fails, do not block; process_upload may still validate.
            pass

    try:
        stats = process_upload(session_path)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    save_json(session_id, "processed/stats.json", stats)

    # Persist any files written by process_upload() when using the GCP storage backend.
    sync_session_tree(session_id)

    return jsonify({
        "session_id": session_id,
        "stats": stats
    }), 200


# -------------------------
# Session status
# -------------------------
@upload_bp.route("/api/sessions/<session_id>/status", methods=["GET"])
@require_auth
def session_status(session_id):
    deny = _ensure_owner(session_id)
    if deny:
        return deny

    try:
        session_path = get_session_path(session_id)
    except FileNotFoundError:
        return jsonify({"error": "Session not found"}), 404

    uploads_dir = os.path.join(session_path, "uploads")
    uploaded_files = {
        ft: os.path.isfile(os.path.join(uploads_dir, f"{ft}.csv"))
        for ft in ("identities", "assignments", "entitlements")
    }

    # Backward compatible: keep boolean uploaded_files, but also include
    # computed metadata for each uploaded CSV (size + row count).
    uploaded_files_info = {
        ft: _upload_file_info(uploads_dir, ft)
        for ft in ("identities", "assignments", "entitlements")
    }

    stats = None
    stats_path = os.path.join(session_path, "processed", "stats.json")
    if os.path.isfile(stats_path):
        try:
            with open(stats_path) as f:
                stats = json.load(f)
        except Exception:
            pass

    has_config = os.path.isfile(os.path.join(session_path, "config.json"))
    has_results = os.path.isfile(os.path.join(session_path, "results", "draft_results.json"))

    return jsonify({
        "session_id": session_id,
        "uploaded_files": uploaded_files,
        "uploaded_files_info": uploaded_files_info,
        "has_processed": stats is not None,
        "stats": stats,
        "has_config": has_config,
        "has_results": has_results,
    }), 200
