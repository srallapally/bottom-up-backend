# routes/browse.py
"""
Serve processed CSV data for the frontend browse views.
Returns JSON arrays parsed from the session's processed CSVs.
"""

import os

import pandas as pd
from flask import Blueprint, jsonify, request

from models.session import get_session_path

browse_bp = Blueprint("browse", __name__)

ALLOWED_FILES = {
    "identities": "identities.csv",
    "assignments": "assignments.csv",
    "entitlements": "catalog.csv",
}

# Hard safety rails: browsing is intended to be paginated. Reading entire CSVs into memory
# is an easy DoS vector once session sizes grow.
MAX_LIMIT = 1000


@browse_bp.route("/api/sessions/<session_id>/browse/<file_type>", methods=["GET"])
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

    # Optional: include scored assignments if available
    if file_type == "assignments":
        scored_path = os.path.join(session_path, "results", "assignments_scored.csv")
        if os.path.isfile(scored_path):
            csv_path = scored_path

    # Pagination is REQUIRED (prevents loading entire CSV into memory).
    limit = request.args.get("limit", type=int)
    offset = request.args.get("offset", 0, type=int)

    if limit is None:
        return jsonify({"error": "limit query param is required"}), 400
    if limit <= 0:
        return jsonify({"error": "limit must be a positive integer"}), 400
    if limit > MAX_LIMIT:
        return jsonify({"error": f"limit must be <= {MAX_LIMIT}"}), 400
    if offset < 0:
        return jsonify({"error": "offset must be >= 0"}), 400

    # Read just the header for columns.
    columns = list(pd.read_csv(csv_path, nrows=0).columns)

    # Compute total rows without parsing the whole file.
    # (lines - 1 header); safe fallback if file is empty.
    with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
        total = max(sum(1 for _ in f) - 1, 0)

    # Read only the requested page.
    # Skip the header row (row 0) plus the requested offset.
    skiprows = range(1, offset + 1) if offset else None
    df = pd.read_csv(csv_path, skiprows=skiprows, nrows=limit)

    return jsonify({
        "file_type": file_type,
        "total": total,
        "offset": offset,
        "count": len(df),
        "columns": columns,
        "rows": df.fillna("").to_dict(orient="records"),
    })