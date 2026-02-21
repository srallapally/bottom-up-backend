"""
Over-Provisioned Access API
============================

Endpoint for retrieving access accumulation / revocation candidates.
"""

import os

import pandas as pd
from flask import Blueprint, jsonify, request

from models.session import get_session_path
from services.auth import require_auth

over_provisioned_bp = Blueprint("over_provisioned", __name__)


@over_provisioned_bp.route(
    "/api/sessions/<session_id>/over-provisioned", methods=["GET"]
)
@require_auth
def get_over_provisioned(session_id):
    """
    GET /api/sessions/<id>/over-provisioned
    Optional query params:
        ?user_id=USR_001   — filter to a specific user
    """
    try:
        session_path = get_session_path(session_id)
    except FileNotFoundError:
        return jsonify({"error": "Session not found"}), 404

    csv_path = os.path.join(session_path, "results", "over_provisioned.csv")
    if not os.path.isfile(csv_path):
        return jsonify({"error": "No results. Run /mine first."}), 404

    df = pd.read_csv(csv_path, dtype=str)

    # Optional user filter
    user_id = request.args.get("user_id")
    if user_id:
        df = df[df["USR_ID"] == user_id]

    records = df.to_dict(orient="records")
    return jsonify({
        "session_id": session_id,
        "count": len(records),
        "over_provisioned": records,
    })