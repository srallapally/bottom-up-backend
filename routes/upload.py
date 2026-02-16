import json
import os

from flask import Blueprint, request, jsonify

from models.session import create_session, get_session_path, save_json, list_sessions
from services.data_loader import process_upload

upload_bp = Blueprint("upload", __name__)

VALID_FILE_TYPES = {"identities", "assignments", "entitlements"}


@upload_bp.route("/api/sessions", methods=["GET"])
def get_sessions():
    sessions = list_sessions()
    return jsonify({"sessions": sessions}), 200


@upload_bp.route("/api/sessions", methods=["POST"])
def new_session():
    session_id = create_session()
    return jsonify({"session_id": session_id}), 201


@upload_bp.route("/api/sessions/<session_id>/upload", methods=["POST"])
def upload_file(session_id):
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

    # Save as canonical name regardless of original filename
    dest = os.path.join(session_path, "uploads", f"{file_type}.csv")
    f.save(dest)

    return jsonify({"session_id": session_id, "file_type": file_type, "status": "uploaded"}), 200


@upload_bp.route("/api/sessions/<session_id>/process", methods=["POST"])
def process_files(session_id):
    try:
        session_path = get_session_path(session_id)
    except FileNotFoundError:
        return jsonify({"error": "Session not found"}), 404

    try:
        stats = process_upload(session_path)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    save_json(session_id, "processed/stats.json", stats)
    return jsonify({"session_id": session_id, "stats": stats}), 200


@upload_bp.route("/api/sessions/<session_id>/status", methods=["GET"])
def session_status(session_id):
    try:
        session_path = get_session_path(session_id)
    except FileNotFoundError:
        return jsonify({"error": "Session not found"}), 404

    uploads_dir = os.path.join(session_path, "uploads")
    uploaded_files = {
        ft: os.path.isfile(os.path.join(uploads_dir, f"{ft}.csv"))
        for ft in ("identities", "assignments", "entitlements")
    }

    # Processed stats
    stats = None
    stats_path = os.path.join(session_path, "processed", "stats.json")
    if os.path.isfile(stats_path):
        try:
            with open(stats_path) as f:
                stats = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    # Config saved?
    has_config = os.path.isfile(os.path.join(session_path, "config.json"))

    # Mining results exist?
    has_results = os.path.isfile(
        os.path.join(session_path, "results", "draft_results.json")
    )

    return jsonify({
        "session_id": session_id,
        "uploaded_files": uploaded_files,
        "has_processed": stats is not None,
        "stats": stats,
        "has_config": has_config,
        "has_results": has_results,
    }), 200