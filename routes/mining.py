"""
Mining Routes - Hybrid Role Mining with Daily Reclustering
==============================================================

New endpoints for V2 hybrid approach:
- POST /api/sessions/<id>/mine              - Initial bootstrap clustering
- GET  /api/sessions/<id>/config            - Get V2 configuration
- PUT  /api/sessions/<id>/config            - Update V2 configuration
- GET  /api/sessions/<id>/results           - Get V2 mining results
- GET  /api/sessions/<id>/draft-roles          - Get draft roles for review
- POST /api/sessions/<id>/draft-roles/<id>/approve - Approve draft role

V1 endpoints remain unchanged in routes/mining.py
"""

import os
import sys
import logging
from typing import Dict, Any

from flask import Blueprint, request, jsonify


logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    MiningConfig,
    DEFAULT_MINING_CONFIG,
    get_default_config,
    merge_configs,
    load_session_config,
    save_session_config,
    initialize_session_directories,
    get_results_path,
)
from models.session import (
    get_session_path,
    save_json,
    load_json,
    load_dataframe,
)
from services.birthright import detect_birthright
from services.clustering import cluster_entitlements_leiden
from services.role_builder import build_roles

try:
    from services.confidence_scorer import (
        score_assignments,
        generate_recommendations,
        detect_over_provisioned,
        build_scoring_summary,
        save_cluster_assignments,
    )
    CONFIDENCE_SCORER_V2_AVAILABLE = True
except ImportError:
    CONFIDENCE_SCORER_V2_AVAILABLE = False


mining_bp = Blueprint("mining", __name__)


# ============================================================================
# CONFIGURATION ENDPOINTS
# ============================================================================

@mining_bp.route("/api/sessions/<session_id>/config", methods=["GET"])
def get_config_v2(session_id):
    """
    GET /api/sessions/<id>/config-v2

    Returns V2 configuration for this session.
    Falls back to V1 config if V2 doesn't exist, then to defaults.
    """
    try:
        session_path = get_session_path(session_id)
    except FileNotFoundError:
        return jsonify({"error": "Session not found"}), 404

    config = load_session_config(session_path)

    return jsonify(config.to_dict()), 200


@mining_bp.route("/api/sessions/<session_id>/config", methods=["PUT"])
def save_config_v2(session_id):
    """
    PUT /api/sessions/<id>/config-v2

    Save V2 configuration for this session.
    Validates before saving.

    Request body: Partial or full config dict (merged with defaults)
    """
    try:
        session_path = get_session_path(session_id)
    except FileNotFoundError:
        return jsonify({"error": "Session not found"}), 404

    # Merge request body with defaults
    body = request.get_json(silent=True) or {}
    config_dict = merge_configs(DEFAULT_MINING_CONFIG, body)
    config = MiningConfig.from_dict(config_dict)

    # Validate
    errors = config.validate()
    if errors:
        return jsonify({
            "error": "Invalid configuration",
            "validation_errors": errors,
        }), 400

    # Save
    save_session_config(session_path, config)

    return jsonify(config.to_dict()), 200


# ============================================================================
# BOOTSTRAP MINING (Initial Role Discovery)
# ============================================================================

@mining_bp.route("/api/sessions/<session_id>/mine", methods=["POST"])
def mine_v2(session_id):
    """
    POST /api/sessions/<id>/mine-v2

    V2 hybrid role mining - bootstrap phase.

    Workflow:
    1. Load processed data (matrix, identities, catalog)
    2. Merge config (defaults + request body overrides)
    3. Validate config
    4. Detect birthright entitlements (reuse V1)
    5. Cluster entitlements using Leiden (V2)
    6. Build draft roles from clusters (V2)
    7. Score assignments (V2 - enhanced)
    8. Generate recommendations (V1 logic)
    9. Save results to results_v2/

    Request body: Optional config overrides
    Response: Draft roles for stakeholder review
    """
    try:
        session_path = get_session_path(session_id)
    except FileNotFoundError:
        return jsonify({"error": "Session not found"}), 404

    # Check processed data exists
    matrix_path = os.path.join(session_path, "processed", "matrix.csv")
    if not os.path.isfile(matrix_path):
        return jsonify({"error": "No processed data. Call /process first."}), 400

    # Initialize V2 directories
    initialize_session_directories(session_path)

    logger.info(f"Starting mining for session {session_id}")

    # Step 1: Load data
    logger.info("Step 1: Loading processed data")
    matrix = load_dataframe(session_id, "processed/matrix.csv")
    identities = load_dataframe(session_id, "processed/identities.csv")

    catalog = None
    catalog_path = os.path.join(session_path, "processed", "catalog.csv")
    if os.path.isfile(catalog_path):
        catalog = load_dataframe(session_id, "processed/catalog.csv")
        logger.info(f"Loaded catalog with {len(catalog)} entitlements")

    logger.info(f"Loaded matrix: {matrix.shape[0]} users × {matrix.shape[1]} entitlements")

    # Step 2: Merge config (defaults + request body)
    logger.info("Step 2: Merging configuration")
    request_body = request.get_json(silent=True) or {}
    config_dict = merge_configs(DEFAULT_MINING_CONFIG, request_body)
    config = MiningConfig.from_dict(config_dict)

    # Step 3: Validate config
    logger.info("Step 3: Validating configuration")
    validation_errors = config.validate()
    if validation_errors:
        return jsonify({
            "error": "Invalid configuration",
            "validation_errors": validation_errors,
        }), 400

    # Save config for reproducibility
    save_session_config(session_path, config)

    # Flatten per-app birthright dict to namespaced list
    birthright_explicit_flat = []
    for app_name, ent_ids in config.birthright_explicit.items():
        for eid in ent_ids:
            birthright_explicit_flat.append(f"{app_name}:{eid}")

    # Step 4: Birthright detection (reuse V1)
    logger.info("Step 4: Detecting birthright entitlements")
    birthright_result = detect_birthright(
        matrix=matrix,
        threshold=config.birthright_threshold,
        explicit_list=birthright_explicit_flat,
        min_assignment_count=config.min_assignment_count,
    )

    logger.info(
        f"Birthright detection complete: "
        f"{len(birthright_result['birthright_entitlements'])} birthright, "
        f"{len(birthright_result.get('noise_entitlements', []))} noise"
    )

    # Step 5: Entitlement clustering (V2 - Leiden)
    logger.info("Step 5: Running Leiden clustering")
    try:
        cluster_result = cluster_entitlements_leiden(
            matrix=birthright_result["filtered_matrix"],
            leiden_min_similarity=config.leiden_min_similarity,
            leiden_min_shared_users=config.leiden_min_shared_users,
            leiden_resolution=config.leiden_resolution,
            leiden_random_seed=config.leiden_random_seed,
            min_entitlement_coverage=config.min_entitlement_coverage,
            max_clusters_per_user=config.max_clusters_per_user,
            min_role_size=config.min_role_size,
            use_sparse=config.use_sparse_matrices,
        )
    except Exception as e:
        return jsonify({
            "error": "Clustering failed",
            "details": str(e),
        }), 500

    logger.info(
        f"Leiden clustering complete: {cluster_result['n_clusters']} clusters, "
        f"modularity={cluster_result['leiden_stats']['modularity']:.3f}"
    )

    # Step 6: Build roles from clusters (V2)
    logger.info("Step 6: Building roles from clusters")
    roles_result = build_roles(
            matrix=matrix,
            cluster_result=cluster_result,
            birthright_result=birthright_result,
            identities=identities,
            catalog=catalog,
            config=config.to_dict(),
    )

    draft_roles = roles_result["roles"]
    birthright_role = roles_result["birthright_role"]
    residuals = roles_result["residuals"]
    multi_cluster_info = roles_result["multi_cluster_info"]
    naming_summary = roles_result["naming_summary"]
    summary_base = roles_result["summary"]

    # Step 7: Build results structure
    results = {
        "roles": draft_roles,
        "birthright_role": birthright_role,
        "residuals": residuals,
        "cluster_info": {
            "leiden_stats": cluster_result["leiden_stats"],
            "n_clusters": cluster_result["n_clusters"],
            "unassigned_users": cluster_result["unassigned_users"],
        },
        "summary": summary_base,
        "multi_cluster_info": multi_cluster_info,
        "naming_summary": naming_summary,
        "config": config.to_dict(),
        "status": "draft",  # Not yet approved by stakeholders
    }

    # Step 8: Confidence Scoring (V2 - Multi-factor)
    logger.info("Step 8: Computing multi-factor confidence scores")

    # Load assignments
    assignments_path = os.path.join(session_path, "processed", "assignments.csv")
    if os.path.isfile(assignments_path):
        assignments_df = load_dataframe(session_id, "processed/assignments.csv").reset_index(drop=True)

        if CONFIDENCE_SCORER_V2_AVAILABLE:
            try:
                # Score all assignments with V2 multi-factor confidence
                enriched_assignments = score_assignments(
                    assignments_df=assignments_df,
                    full_matrix=matrix,
                    identities=identities,
                    cluster_result=cluster_result,
                    roles=draft_roles,
                    birthright_entitlements=birthright_result["birthright_entitlements"],
                    noise_entitlements=birthright_result.get("noise_entitlements", []),
                    config=config.to_dict(),
                    drift_data=None,  # TODO: Load from daily clustering when available
                )

                # Save enriched assignments
                # enriched_assignments.to_csv(assignments_path, index=False)
                results_path = get_results_path(session_path)
                enriched_assignments.to_csv(
                    os.path.join(results_path, "assignments_scored.csv"),
                    index=False
                )
                # Generate recommendations (missing entitlements with high confidence)
                # TODO: Implement recommendations_v2 properly
                recommendations = generate_recommendations(
                    enriched_assignments=enriched_assignments,
                    full_matrix=matrix,
                    cluster_result=cluster_result,
                    config=config.to_dict(),
                )

                # Detect over-provisioned access (low confidence)
                over_provisioned = detect_over_provisioned(
                    enriched_assignments=enriched_assignments,
                    revocation_threshold=config.revocation_threshold,
                )

                # results_path = get_results_path(session_path)

                # Save scoring outputs to results_v2/
                recommendations.to_csv(
                    os.path.join(results_path, "recommendations.csv"),
                    index=False
                )
                over_provisioned.to_csv(
                    os.path.join(results_path, "over_provisioned.csv"),
                    index=False
                )

                # Build scoring summary
                scoring_summary = build_scoring_summary(
                    enriched_assignments=enriched_assignments,
                    cluster_result=cluster_result,
                    birthright_entitlements=birthright_result["birthright_entitlements"],
                )

                # Add to results
                results.update(scoring_summary)

                logger.info(
                    f"Confidence scoring complete: "
                    f"{scoring_summary['confidence_scoring']['high']} HIGH, "
                    f"{scoring_summary['confidence_scoring']['medium']} MEDIUM, "
                    f"{scoring_summary['confidence_scoring']['low']} LOW"
                )

            except Exception as e:
                logger.error(f"Confidence scoring failed: {e}", exc_info=True)
                # Continue without scoring
                results["confidence_scoring"] = {
                    "error": str(e),
                    "message": "Confidence scoring failed, results available without scoring"
                }
        else:
            logger.warning("Confidence scorer V2 not available, skipping")
            results["confidence_scoring"] = {
                "error": "confidence_scorer_v2 not installed",
                "message": "Install services/confidence_scorer.py for scoring"
            }
    else:
        logger.warning("No assignments.csv found, skipping confidence scoring")

    # Step 9: Save results to results/
    results_path = get_results_path(session_path)
    save_json(session_id, "results/draft_results.json", results)

    # Save cluster assignments for future drift detection
    # (Store user_cluster_membership for daily comparison)
    save_json(
        session_id,
        "results/cluster_membership.json",
        cluster_result["user_cluster_membership"],
    )

    return jsonify(results), 200


# ============================================================================
# DRAFT ROLE MANAGEMENT
# ============================================================================

@mining_bp.route("/api/sessions/<session_id>/draft-roles", methods=["GET"])
def get_draft_roles(session_id):
    """
    GET /api/sessions/<id>/draft-roles

    Returns draft roles for stakeholder review.
    """
    try:
        session_path = get_session_path(session_id)
    except FileNotFoundError:
        return jsonify({"error": "Session not found"}), 404

    try:
        draft_results = load_json(session_id, "results/draft_results.json")
    except FileNotFoundError:
        return jsonify({"error": "No draft roles. Run /mine first."}), 404

    return jsonify({
        "roles": draft_results["roles"],
        "summary": draft_results["summary"],
        "cluster_info": draft_results["cluster_info"],
    }), 200


@mining_bp.route("/api/sessions/<session_id>/draft-roles/<role_id>/approve", methods=["POST"])
def approve_draft_role(session_id, role_id):
    """
    POST /api/sessions/<id>/draft-roles/<role_id>/approve

    Approve a draft role (converts to business role).

    Request body:
    {
        "role_name": "Cloud Infrastructure Engineer",  // Optional override
        "owner": "CTO",                                // Required
        "purpose": "Manage cloud infrastructure",      // Optional
        "entitlements": ["AWS:S3", "AWS:EC2"],        // Optional override
    }

    TODO: Implement business role versioning and storage
    For now, just acknowledges the approval.
    """
    try:
        session_path = get_session_path(session_id)
    except FileNotFoundError:
        return jsonify({"error": "Session not found"}), 404

    try:
        draft_results = load_json(session_id, "results/draft_results.json")
    except FileNotFoundError:
        return jsonify({"error": "No draft roles. Run /mine first."}), 404

    # Find the role
    role = None
    for r in draft_results["roles"]:
        if r["role_id"] == role_id:
            role = r
            break

    if not role:
        return jsonify({"error": f"Role {role_id} not found"}), 404

    # Get approval details from request
    body = request.get_json(silent=True) or {}

    # Validate required fields
    if "owner" not in body:
        return jsonify({"error": "Field 'owner' is required"}), 400

    # Build business role
    business_role = {
        "role_id": role_id,
        "role_name": body.get("role_name", role.get("role_name", role_id)),
        "owner": body["owner"],
        "purpose": body.get("purpose", ""),
        "entitlements": body.get("entitlements", role["entitlements"]),
        "members": role["members"],
        "member_coverage": role.get("member_coverage", {}),
        "version": "1.0",
        "status": "ACTIVE",
        "source": "draft_approval",
        "approved_by": body.get("approved_by", "user"),
        "approved_at": _get_current_timestamp(),
    }

    # Save to business_roles/
    business_roles_dir = os.path.join(session_path, "business_roles")
    os.makedirs(business_roles_dir, exist_ok=True)

    save_json(
        session_id,
        f"business_roles/{role_id}.json",
        business_role,
    )

    return jsonify({
        "message": f"Role {role_id} approved",
        "business_role": business_role,
    }), 200


# ============================================================================
# RESULTS RETRIEVAL
# ============================================================================

@mining_bp.route("/api/sessions/<session_id>/results", methods=["GET"])
def get_results_v2(session_id):
    """
    GET /api/sessions/<id>/results-v2

    Returns V2 mining results.
    """
    try:
        get_session_path(session_id)
    except FileNotFoundError:
        return jsonify({"error": "Session not found"}), 404

    try:
        results = load_json(session_id, "results/draft_results.json")
    except FileNotFoundError:
        return jsonify({"error": "No results. Run /mine first."}), 404

    return jsonify(results), 200


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _build_draft_roles_from_clusters(
    cluster_result: Dict[str, Any],
    matrix,
    identities,
    catalog,
    config: MiningConfig,
) -> list:
    """
    Build draft roles from entitlement clusters.

    TODO: Replace with role_builder_v2.py when implemented.
    This is a simplified version for now.
    """
    entitlement_clusters = cluster_result["entitlement_clusters"]
    user_memberships = cluster_result["user_cluster_membership"]
    cluster_metadata = cluster_result["cluster_metadata"]

    roles = []

    for cluster_id, cluster_ents in entitlement_clusters.items():
        # Find users in this cluster
        members = [
            user_id for user_id, memberships in user_memberships.items()
            if any(m["cluster_id"] == cluster_id for m in memberships)
        ]

        # Get coverage for each member
        member_coverage = {}
        for user_id in members:
            for m in user_memberships[user_id]:
                if m["cluster_id"] == cluster_id:
                    member_coverage[user_id] = {
                        "coverage": m["coverage"],
                        "has_count": m["count"],
                        "total_count": m["total"],
                    }
                    break

        # Auto-generate role name from HR attributes
        role_name = _generate_role_name(
            members=members,
            identities=identities,
            cluster_id=cluster_id,
            config=config,
        )

        # Get HR summary
        hr_summary = _summarize_hr_attributes(members, identities)

        # Enrich with catalog
        entitlements_detail = []
        if catalog is not None:
            catalog_lookup = {
                row["namespaced_id"]: {
                    "app_id": row["APP_ID"],
                    "ent_name": row.get("ENT_NAME", row["ENT_ID"]),
                }
                for _, row in catalog.iterrows()
            }

            for ent_id in cluster_ents:
                detail = catalog_lookup.get(ent_id, {})
                entitlements_detail.append({
                    "entitlement_id": ent_id,
                    "app_id": detail.get("app_id", ent_id.split(":")[0]),
                    "ent_name": detail.get("ent_name", ent_id.split(":")[-1]),
                })
        else:
            for ent_id in cluster_ents:
                parts = ent_id.split(":", 1)
                entitlements_detail.append({
                    "entitlement_id": ent_id,
                    "app_id": parts[0] if len(parts) == 2 else "",
                    "ent_name": parts[1] if len(parts) == 2 else ent_id,
                })

        meta = cluster_metadata.get(cluster_id, {})

        role = {
            "role_id": f"ROLE_{cluster_id:03d}",
            "role_name": role_name,
            "entitlement_cluster_id": cluster_id,
            "entitlements": cluster_ents,
            "entitlement_count": len(cluster_ents),
            "members": members,
            "member_count": len(members),
            "member_coverage": member_coverage,
            "avg_coverage": meta.get("avg_coverage", 0.0),
            "hr_summary": hr_summary,
            "entitlements_detail": entitlements_detail,
            "status": "draft",
        }

        roles.append(role)

    return roles


def _generate_role_name(
    members: list,
    identities,
    cluster_id: int,
    config: MiningConfig,
) -> str:
    """
    Auto-generate role name from HR attribute dominance.

    Returns semantic name like "Engineering - DevOps Engineer"
    or numeric fallback like "ROLE_007"
    """
    if not config.auto_generate_role_names or not members:
        return f"ROLE_{cluster_id:03d}"

    # Get HR data for members
    member_ids = [m for m in members if m in identities.index]
    if not member_ids:
        return f"ROLE_{cluster_id:03d}"

    member_data = identities.loc[member_ids]

    primary_attr = config.role_name_primary_attr
    secondary_attr = config.role_name_secondary_attr
    min_dominance = config.role_name_min_dominance

    # Check primary attribute
    if primary_attr in member_data.columns:
        primary_counts = member_data[primary_attr].value_counts()
        if not primary_counts.empty:
            top_primary = primary_counts.index[0]
            top_primary_pct = primary_counts.iloc[0] / len(member_data)

            if top_primary_pct >= min_dominance:
                # Check secondary
                if secondary_attr in member_data.columns:
                    secondary_counts = member_data[secondary_attr].value_counts()
                    if not secondary_counts.empty:
                        top_secondary = secondary_counts.index[0]
                        top_secondary_pct = secondary_counts.iloc[0] / len(member_data)

                        if top_secondary_pct >= min_dominance:
                            return f"{top_primary} - {top_secondary}"

                return str(top_primary)

    # Fallback: numeric
    return f"ROLE_{cluster_id:03d}"


def _summarize_hr_attributes(members: list, identities) -> dict:
    """Get top HR attribute values for role members."""
    member_ids = [m for m in members if m in identities.index]
    if not member_ids:
        return {}

    member_data = identities.loc[member_ids]

    hr_cols = ["department", "business_unit", "jobcode", "job_level", "location_country"]
    summary = {}

    for col in hr_cols:
        if col not in member_data.columns:
            continue

        counts = member_data[col].value_counts().head(3)
        summary[col] = [
            {
                "value": str(v),
                "count": int(c),
                "pct": round(c / len(member_data), 4),
            }
            for v, c in counts.items()
        ]

    return summary


def _build_birthright_role(birthright_result: dict, catalog) -> dict:
    """Build birthright role structure."""
    birthright_ents = birthright_result["birthright_entitlements"]
    birthright_stats = birthright_result["birthright_stats"]

    # Enrich with catalog
    entitlements_detail = []
    if catalog is not None:
        catalog_lookup = {
            row["namespaced_id"]: {
                "app_id": row["APP_ID"],
                "ent_name": row.get("ENT_NAME", row["ENT_ID"]),
            }
            for _, row in catalog.iterrows()
        }

        for ent_id in birthright_ents:
            detail = catalog_lookup.get(ent_id, {})
            entitlements_detail.append({
                "entitlement_id": ent_id,
                "app_id": detail.get("app_id", ent_id.split(":")[0]),
                "ent_name": detail.get("ent_name", ent_id.split(":")[-1]),
            })
    else:
        for ent_id in birthright_ents:
            parts = ent_id.split(":", 1)
            entitlements_detail.append({
                "entitlement_id": ent_id,
                "app_id": parts[0] if len(parts) == 2 else "",
                "ent_name": parts[1] if len(parts) == 2 else ent_id,
            })

    return {
        "role_id": "ROLE_BIRTHRIGHT",
        "role_name": "Organization-Wide Baseline Access",
        "entitlements": birthright_ents,
        "entitlement_count": len(birthright_ents),
        "stats": birthright_stats,
        "entitlements_detail": entitlements_detail,
    }


def _build_summary(cluster_result: dict, birthright_result: dict, matrix) -> dict:
    """Build summary statistics."""
    n_clusters = cluster_result["n_clusters"]
    user_memberships = cluster_result["user_cluster_membership"]

    # Multi-cluster statistics
    multi_cluster_count = sum(
        1 for memberships in user_memberships.values()
        if len(memberships) > 1
    )

    cluster_counts = {}
    for memberships in user_memberships.values():
        count = len(memberships)
        cluster_counts[count] = cluster_counts.get(count, 0) + 1

    total_assigned = len(user_memberships)
    total_users = len(matrix)

    return {
        "total_roles": n_clusters,
        "total_users": total_users,
        "assigned_users": total_assigned,
        "unassigned_users": total_users - total_assigned,
        "multi_cluster_users": multi_cluster_count,
        "single_cluster_users": total_assigned - multi_cluster_count,
        "avg_clusters_per_user": round(
            sum(len(m) for m in user_memberships.values()) / total_assigned, 2
        ) if total_assigned > 0 else 0.0,
        "birthright_entitlements": len(birthright_result["birthright_entitlements"]),
        "noise_entitlements": len(birthright_result["noise_entitlements"]),
        "multi_cluster_distribution": cluster_counts,
    }


def _get_current_timestamp() -> str:
    """Get current timestamp in ISO format."""
    from datetime import datetime
    return datetime.utcnow().isoformat() + "Z"