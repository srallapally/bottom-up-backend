# routes/mining.py
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

from flask import Blueprint, request, jsonify, g
logger = logging.getLogger(__name__)

from services.auth import require_auth
from services.cloud_run_jobs import run_mine_job


def _ensure_owner(session_id: str):
    owner = session_owner_sub(session_id)
    if not owner:
        return jsonify({"error": "Session missing owner metadata"}), 403
    if owner != g.user["sub"]:
        return jsonify({"error": "Forbidden"}), 403
    return None


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
    session_owner_sub,
    update_session_fields,
    sync_session_tree,
)
from services.birthright import detect_birthright
from services.clustering import cluster_entitlements_leiden
from services.role_builder import build_roles

# CHANGE 2026-02-17: Import for sparse matrix loading
import scipy.sparse as sp
import pandas as pd


def load_sparse_matrix(session_id: str):
    """
    CHANGE 2026-02-17: Load sparse matrix from npz file with indices.

    Returns:
        tuple: (sparse_matrix, user_ids, ent_ids)
    """
    session_path = get_session_path(session_id)
    matrix_path = os.path.join(session_path, "processed", "matrix.npz")
    users_path = os.path.join(session_path, "processed", "matrix_users.csv")
    ents_path = os.path.join(session_path, "processed", "matrix_entitlements.csv")

    matrix_sparse = sp.load_npz(matrix_path)
    user_ids = pd.read_csv(users_path)["USR_ID"].values
    ent_ids = pd.read_csv(ents_path)["namespaced_id"].values

    return matrix_sparse, user_ids, ent_ids


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

@require_auth
def get_config_v2(session_id):
    """
    GET /api/sessions/<id>/config-v2

    Returns V2 configuration for this session.
    Falls back to V1 config if V2 doesn't exist, then to defaults.
    """
    deny = _ensure_owner(session_id)
    if deny:
        return deny

    try:
        session_path = get_session_path(session_id)
    except FileNotFoundError:
        return jsonify({"error": "Session not found"}), 404

    config = load_session_config(session_path)

    return jsonify(config.to_dict()), 200


@mining_bp.route("/api/sessions/<session_id>/config", methods=["PUT"])

@require_auth
def save_config_v2(session_id):
    """
    PUT /api/sessions/<id>/config-v2

    Save V2 configuration for this session.
    Validates before saving.

    Request body: Partial or full config dict (merged with defaults)
    """
    deny = _ensure_owner(session_id)
    if deny:
        return deny

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

@require_auth
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
    deny = _ensure_owner(session_id)
    if deny:
        return deny

    try:
        session_path = get_session_path(session_id)
    except FileNotFoundError:
        return jsonify({"error": "Session not found"}), 404

    # Check processed data exists
    # CHANGE 2026-02-17: Now checking for .npz file instead of .csv
    matrix_path = os.path.join(session_path, "processed", "matrix.npz")
    if not os.path.isfile(matrix_path):
        return jsonify({"error": "No processed data. Call /process first."}), 400

    execution_backend = os.getenv("EXECUTION_BACKEND", "local").strip().lower()
    if execution_backend in {"gcp", "cloud", "cloud_run", "cloud_run_jobs", "jobs"}:
        # Trigger async mining via Cloud Run Jobs. Endpoint contract unchanged.
        # IMPORTANT: Persist any request-body config overrides before triggering the job,
        # so the worker reads the correct config.json.
        try:
            request_body = request.get_json(silent=True) or {}
            if request_body:
                config_dict = merge_configs(DEFAULT_MINING_CONFIG, request_body)
                config = MiningConfig.from_dict(config_dict)

                validation_errors = config.validate()
                if validation_errors:
                    return jsonify({
                        "error": "Invalid configuration",
                        "validation_errors": validation_errors,
                    }), 400

                identities = load_dataframe(session_id, "processed/identities.csv")
                data_validation_errors = config.validate_against_data(
                    identities_columns=list(identities.columns)
                )
                if data_validation_errors:
                    return jsonify({
                        "error": "Mining configuration error: attribute columns not found in data",
                        "validation_errors": data_validation_errors,
                    }), 400

                save_session_config(session_path, config)

            update_session_fields(session_id, {"status": "MINING"})
            op = run_mine_job(session_id=session_id, owner_sub=g.user["sub"])
            return jsonify({
                "session_id": session_id,
                "status": "MINING",
                "operation": op.get("name"),
            }), 202
        except Exception as e:
            try:
                update_session_fields(session_id, {"status": "FAILED", "error": str(e)})
            except Exception:
                pass
            return jsonify({"error": str(e)}), 500


    # Initialize V2 directories
    initialize_session_directories(session_path)

    logger.info(f"Starting mining for session {session_id}")

    # Step 1: Load data
    logger.info("Step 1: Loading processed data")
    # CHANGE 2026-02-17: Load sparse matrix + indices instead of dense DataFrame
    matrix, user_ids, ent_ids = load_sparse_matrix(session_id)
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

    # ADDED: Validate that configured user_attributes columns exist in identities.csv
    # Hard fail if any column is missing — customer must fix config before mining.
    data_validation_errors = config.validate_against_data(
        identities_columns=list(identities.columns)
    )
    if data_validation_errors:
        return jsonify({
            "error": "Mining configuration error: attribute columns not found in data",
            "validation_errors": data_validation_errors,
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
    # CHANGE 2026-02-17: Pass sparse matrix with indices
    birthright_result = detect_birthright(
        matrix=matrix,
        user_ids=user_ids,
        ent_ids=ent_ids,
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
        # CHANGE 2026-02-17: Pass filtered_ent_ids from birthright result
        cluster_result = cluster_entitlements_leiden(
            matrix=birthright_result["filtered_matrix"],
            ent_ids=birthright_result["filtered_ent_ids"],
            user_ids=user_ids,
            leiden_min_similarity=config.leiden_min_similarity,
            leiden_min_shared_users=config.leiden_min_shared_users,
            leiden_resolution=config.leiden_resolution,
            leiden_random_seed=config.leiden_random_seed,
            min_entitlement_coverage=config.min_entitlement_coverage,
            min_absolute_overlap=config.min_absolute_overlap,
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
        f"cpm_quality={cluster_result['leiden_stats']['cpm_quality']:.3f}"
    )

    # Step 6: Build roles from clusters (V2)
    logger.info("Step 6: Building roles from clusters")
    # CHANGE 2026-02-17: Pass user_ids and ent_ids for sparse matrix compatibility
    roles_result = build_roles(
        matrix=matrix,
        user_ids=user_ids,
        ent_ids=ent_ids,
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
    birthright_promotions = roles_result.get("birthright_promotions", [])
    merge_candidates = roles_result.get("merge_candidates", [])

    # Step 7: Build results structure
    results = {
        "roles": draft_roles,
        "birthright_role": birthright_role,
        "birthright_promotions": birthright_promotions,
        "merge_candidates": merge_candidates,
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
                from services.confidence_scorer import SparseMatrixAccessor
                matrix_for_scoring = SparseMatrixAccessor(matrix, user_ids, ent_ids)

                # Score all assignments with V2 multi-factor confidence
                enriched_assignments = score_assignments(
                    assignments_df=assignments_df,
                    full_matrix=matrix_for_scoring,
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
                    # CHANGE 2026-02-17: Pass SparseMatrixAccessor for sparse-safe lookups
                    # (avoids accidental densification inside recommendation logic)
                    full_matrix=matrix_for_scoring,
                    cluster_result=cluster_result,
                    config=config.to_dict(),
                )

                # Detect over-provisioned access (low confidence)
                over_provisioned = detect_over_provisioned(
                    enriched_assignments=enriched_assignments,
                    revocation_threshold=config.revocation_threshold,
                )

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

    update_session_fields(session_id, {"status": "MINED"})
    sync_session_tree(session_id)


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

@require_auth
def get_draft_roles(session_id):
    """
    GET /api/sessions/<id>/draft-roles

    Returns draft roles for stakeholder review.
    """
    deny = _ensure_owner(session_id)
    if deny:
        return deny

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

@require_auth
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
    deny = _ensure_owner(session_id)
    if deny:
        return deny

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

@require_auth
def get_results_v2(session_id):
    """
    GET /api/sessions/<id>/results-v2

    Returns V2 mining results.
    """
    deny = _ensure_owner(session_id)
    if deny:
        return deny

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
# CHANGED: Removed _build_draft_roles_from_clusters, _generate_role_name,
# _summarize_hr_attributes, _build_birthright_role.
# These were duplicates of logic now in services/role_builder.py.
# The mine_v2 route calls build_roles() which handles all role construction,
# naming (ROLE_NNN), HR summary (from config user_attributes), and birthright.


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
