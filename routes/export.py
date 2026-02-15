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
        results = load_json(session_id, "results/results.json")
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
            "config": results["config"],
            "summary": results["summary"],
            "cluster_info": results["cluster_info"],
            "confidence_scoring": results.get("confidence_scoring", {}),
        }, indent=2))

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
    headers = ["ROLE_ID", "APP_ID", "ENT_ID", "ENT_NAME"]
    rows = []
    for role in roles:
        for ent in role.get("entitlements_detail", []):
            rows.append([
                role["role_id"],
                ent["app_id"],
                ent["entitlement_id"],
                ent["ent_name"],
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