import os

import pandas as pd
from flask import Blueprint, jsonify, request

from models.session import get_session_path

assignments_bp = Blueprint("assignments", __name__)

@assignments_bp.route("/api/sessions/<session_id>/assignments", methods=["GET"])
def assignments(session_id):
    try:
        session_path = get_session_path(session_id)
    except FileNotFoundError:
        return jsonify({"error": "Session not found"}), 404

    scored_path = os.path.join(session_path, "results", "assignments_scored.csv")
    original_path = os.path.join(session_path, "processed", "assignments.csv")

    has_confidence_scores = False

    if os.path.isfile(scored_path):
        df = pd.read_csv(scored_path, dtype=str)
        has_confidence_scores = True
    elif os.path.isfile(original_path):
        df = pd.read_csv(original_path, dtype=str)
        has_confidence_scores = False
    else:
        return jsonify({"error": "No assignments data. Upload files first."}), 404

    # Step 3: Parse query parameters
    user_id = request.args.get("user_id")
    confidence_level = request.args.get("confidence_level", "").upper()
    limit = request.args.get("limit", type=int)
    include_metadata = request.args.get("include_metadata", "false").lower() == "true"

    # Step 4: Validate confidence_level if provided
    if confidence_level and confidence_level not in ["HIGH", "MEDIUM", "LOW"]:
        return jsonify({"error": "confidence_level must be one of: HIGH, MEDIUM, LOW"}), 400

    # Step 5: Apply filters
    if user_id:
        df = df[df["USR_ID"] == user_id]

    if confidence_level and has_confidence_scores:
        if "confidence_level" in df.columns:
            df = df[df["confidence_level"] == confidence_level]

    # Step 6: Apply limit
    if limit and limit > 0:
        df = df.head(limit)

    # Step 7: Filter columns if metadata not requested
    if not include_metadata and has_confidence_scores:
        # Keep only user-facing columns
        keep_cols = [
            "USR_ID", "APP_ID", "ENT_ID", "namespaced_id",
            "confidence", "confidence_level", "justification"
        ]
        existing_cols = [c for c in keep_cols if c in df.columns]
        df = df[existing_cols]

    # Step 8: Build response
    records = df.to_dict(orient="records")

    response = {
        "session_id": session_id,
        "has_confidence_scores": has_confidence_scores,
        "count": len(records),
        "assignments": records,
    }

    # Add filters applied
    filters_applied = {}
    if user_id:
        filters_applied["user_id"] = user_id
    if confidence_level:
        filters_applied["confidence_level"] = confidence_level
    if limit:
        filters_applied["limit"] = limit

    if filters_applied:
        response["filters_applied"] = filters_applied

    # Add warning if no confidence scores
    if not has_confidence_scores:
        response["warning"] = "Confidence scores not available. Run /mine to generate scores."

    return jsonify(response), 200