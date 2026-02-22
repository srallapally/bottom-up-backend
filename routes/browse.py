# routes/browse.py
"""
Serve processed CSV data for the frontend browse views.
Returns JSON arrays parsed from the session's processed CSVs.
"""

import os

import pandas as pd
from flask import Blueprint, jsonify, request

from models.session import get_session_path, session_owner_sub
from services.auth import require_auth
from flask import g

browse_bp = Blueprint("browse", __name__)

ALLOWED_FILES = {
    "identities": "identities.csv",
    "assignments": "assignments.csv",
    "entitlements": "catalog.csv",
}


@browse_bp.route("/api/sessions/<session_id>/browse/<file_type>", methods=["GET"])
@require_auth
def browse_data(session_id, file_type):
    if file_type not in ALLOWED_FILES:
        return jsonify({"error": f"file_type must be one of: {', '.join(sorted(ALLOWED_FILES))}"}), 400

    try:
        session_path = get_session_path(session_id)
    except FileNotFoundError:
        return jsonify({"error": "Session not found"}), 404

    csv_path = os.path.join(session_path, "processed", ALLOWED_FILES[file_type])
    if not os.path.isfile(csv_path):
        return jsonify({"error": f"Processed {file_type} not found. Run /process first."}), 404

    df = pd.read_csv(csv_path)

    # Optional: include scored assignments if available
    if file_type == "assignments":
        scored_path = os.path.join(session_path, "results", "assignments_scored.csv")
        if os.path.isfile(scored_path):
            df = pd.read_csv(scored_path)

    # Pagination via query params (optional)
    # NOTE: Frontend callers may omit limit; default to a reasonable page size.
    limit = request.args.get("limit", default=100, type=int)
    offset = request.args.get("offset", 0, type=int)

    total = len(df)
    if limit and limit > 0:
        df = df.iloc[offset: offset + limit]

    return jsonify({
        "file_type": file_type,
        "total": total,
        "offset": offset,
        "count": len(df),
        "columns": list(df.columns),
        "rows": df.fillna("").to_dict(orient="records"),
    })


# ---------------------------------------------------------------------------
# Legacy compatibility routes
#
# Some frontend builds call /identities, /entitlements, /assignments (optionally
# under /api) and pass session_id as a query parameter. These endpoints delegate
# to the canonical /api/sessions/<session_id>/browse/<file_type> route.
# ---------------------------------------------------------------------------

@browse_bp.route("/api/<file_type>", methods=["GET"])
@browse_bp.route("/<file_type>", methods=["GET"])
@require_auth
def browse_data_legacy(file_type):
    if file_type not in ALLOWED_FILES:
        return jsonify({"error": f"file_type must be one of: {', '.join(sorted(ALLOWED_FILES))}"}), 400

    session_id = request.args.get("session_id")
    if not session_id:
        return jsonify({"error": "session_id query param is required"}), 400

    # FIX 12: The interceptor's ownership check only fires when session_id is a
    # path variable in request.view_args. The legacy route passes session_id as a
    # query param, so the interceptor silently skips ownership enforcement.
    # Perform it explicitly here.
    owner = session_owner_sub(session_id)
    if not owner:
        return jsonify({"error": "Session missing owner", "userMessage": "Session missing owner"}), 403
    if owner != g.user["sub"]:
        return jsonify({"error": "Forbidden", "userMessage": "Forbidden"}), 403

    # Delegate to the canonical handler.
    return browse_data(session_id=session_id, file_type=file_type)