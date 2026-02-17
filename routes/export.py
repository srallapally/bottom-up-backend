# routes/export.py
import io
import csv
import json
import os
import zipfile

from flask import Blueprint, Response, jsonify

from models.session import get_session_path, load_json

export_bp = Blueprint("export", __name__)


@export_bp.route("/api/sessions/<session_id>/export", methods=["GET"])
def export_results(session_id):
    try:
        session_path = get_session_path(session_id)
    except FileNotFoundError:
        return jsonify({"error": "Session not found"}), 404

    try:
        results = load_json(session_id, "results/draft_results.json")
    except FileNotFoundError:
        return jsonify({"error": "No results. Call /mine first."}), 404

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("roles.csv", _build_roles_csv(results["roles"]))
        zf.writestr("role_members.csv", _build_members_csv(results["roles"]))
        zf.writestr("birthright_role.csv", _build_birthright_csv(results["birthright_role"]))
        zf.writestr("residual_access.csv", _build_residuals_csv(results["residuals"]))

        # Confidence scoring outputs
        recs_path = os.path.join(session_path, "results", "recommendations.csv")
        if os.path.isfile(recs_path):
            with open(recs_path, "r") as f:
                zf.writestr("recommendations.csv", f.read())

        op_path = os.path.join(session_path, "results", "over_provisioned.csv")
        if os.path.isfile(op_path):
            with open(op_path, "r") as f:
                zf.writestr("over_provisioned.csv", f.read())

        zf.writestr("summary.json", json.dumps({
            "config": results.get("config", {}),
            "summary": results.get("summary", {}),
            "cluster_info": results.get("cluster_info", {}),
            "confidence_scoring": results.get("confidence_scoring", {}),
        }, indent=2))

        # Birthright promotion candidates
        promotions = results.get("birthright_promotions", [])
        if promotions:
            zf.writestr("birthright_promotions.csv", _build_promotions_csv(promotions))

        # Merge candidates
        candidates = results.get("merge_candidates", [])
        if candidates:
            zf.writestr("merge_candidates.csv", _build_merge_candidates_csv(candidates))

        # Scored assignments (from confidence scorer)
        scored_path = os.path.join(session_path, "results", "assignments_scored.csv")
        if os.path.isfile(scored_path):
            with open(scored_path, "r") as f:
                zf.writestr("assignments_scored.csv", f.read())

    buf.seek(0)
    return Response(
        buf.getvalue(),
        mimetype="application/zip",
        headers={"Content-Disposition": f"attachment; filename=role_mining_{session_id[:8]}.zip"},
    )


def _to_csv_string(headers: list[str], rows: list[list]) -> str:
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(headers)
    writer.writerows(rows)
    return out.getvalue()


def _build_roles_csv(roles: list[dict]) -> str:
    headers = ["ROLE_ID", "TIER", "APP_ID", "ENT_ID", "ENT_NAME", "PREVALENCE"]
    rows = []
    for role in roles:
        core_set = set(role.get("core_entitlements", []))
        common_set = set(role.get("common_entitlements", []))
        prevalence = role.get("entitlement_prevalence", {})

        for ent in role.get("entitlements_detail", []):
            ent_id = ent["entitlement_id"]
            if ent_id in core_set:
                tier = "core"
            elif ent_id in common_set:
                tier = "common"
            else:
                tier = "seed"
            rows.append([
                role["role_id"],
                tier,
                ent["app_id"],
                ent_id,
                ent["ent_name"],
                prevalence.get(ent_id, ""),
            ])
    return _to_csv_string(headers, rows)


def _build_members_csv(roles: list[dict]) -> str:
    headers = ["ROLE_ID", "USR_ID"]
    rows = []
    for role in roles:
        for user_id in role["members"]:
            rows.append([role["role_id"], user_id])
    return _to_csv_string(headers, rows)


def _build_birthright_csv(birthright_role: dict) -> str:
    headers = ["ROLE_ID", "APP_ID", "ENT_ID", "ENT_NAME", "PCT_USERS"]
    rows = []
    stats = birthright_role.get("stats", {})
    for ent in birthright_role.get("entitlements_detail", []):
        ent_id = ent["entitlement_id"]
        pct = stats.get(ent_id, {}).get("pct", 0)
        rows.append([
            "ROLE_BIRTHRIGHT",
            ent["app_id"],
            ent_id,
            ent["ent_name"],
            round(pct * 100, 2),
        ])
    return _to_csv_string(headers, rows)


def _build_residuals_csv(residuals: list[dict]) -> str:
    headers = ["USR_ID", "ENT_ID"]
    rows = [[r["USR_ID"], r["entitlement_id"]] for r in residuals]
    return _to_csv_string(headers, rows)


def _build_promotions_csv(promotions: list[dict]) -> str:
    headers = ["ENT_ID", "CORE_IN_ROLES", "TOTAL_ROLES", "PCT"]
    rows = [
        [p["entitlement_id"], p["core_in_roles"], p["total_roles"], p["pct"]]
        for p in promotions
    ]
    return _to_csv_string(headers, rows)


def _build_merge_candidates_csv(candidates: list[dict]) -> str:
    headers = ["ROLE_A", "ROLE_B", "JACCARD", "SHARED_CORE_COUNT", "SHARED_CORE"]
    rows = [
        [c["role_a"], c["role_b"], c["jaccard"], c["shared_core_count"],
         "; ".join(c.get("shared_core", []))]
        for c in candidates
    ]
    return _to_csv_string(headers, rows)