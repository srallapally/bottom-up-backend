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

CHANGE 2026-04-02: Added Step 6.5 — hybrid co-occurrence mining on Leiden
residuals. Discovers supplementary roles for unassigned users and merges
them into the roles list before confidence scoring.
"""

import json
import os
import sys
import logging
from typing import Dict, Any

import numpy as np
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


def _sanitize_for_json(obj):
    """
    Recursively convert numpy types to native Python types so json.dumps
    doesn't choke on numpy.int64 keys or numpy.float64 values.
    """
    if isinstance(obj, dict):
        return {_sanitize_for_json(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


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
    sync_file,
    fetch_file,
)
from services.birthright import detect_birthright, detect_tiered_birthrights
from services.clustering import cluster_entitlements_leiden
from services.role_builder import build_roles

# CHANGE 2026-04-02: Import hybrid miner
from services.hybrid_miner import discover_hybrid_roles, merge_hybrid_into_cluster_result

import scipy.sparse as sp
import pandas as pd


def load_sparse_matrix(session_id: str):
    """Load sparse matrix from npz file with indices."""
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
    deny = _ensure_owner(session_id)
    if deny:
        return deny

    try:
        session_path = get_session_path(session_id)
    except FileNotFoundError:
        return jsonify({"error": "Session not found"}), 404

    body = request.get_json(silent=True) or {}
    config_dict = merge_configs(DEFAULT_MINING_CONFIG, body)
    config = MiningConfig.from_dict(config_dict)

    errors = config.validate()
    if errors:
        return jsonify({
            "error": "Invalid configuration",
            "validation_errors": errors,
        }), 400

    save_session_config(session_path, config)
    sync_file(session_id, "config.json")

    return jsonify(config.to_dict()), 200


# ============================================================================
# BOOTSTRAP MINING (Initial Role Discovery)
# ============================================================================

@mining_bp.route("/api/sessions/<session_id>/mine", methods=["POST"])

@require_auth
def mine_v2(session_id):
    deny = _ensure_owner(session_id)
    if deny:
        return deny

    try:
        session_path = get_session_path(session_id)
    except FileNotFoundError:
        return jsonify({"error": "Session not found"}), 404

    matrix_path = os.path.join(session_path, "processed", "matrix.npz")
    if not os.path.isfile(matrix_path):
        return jsonify({"error": "No processed data. Call /process first."}), 400

    execution_backend = os.getenv("EXECUTION_BACKEND", "local").strip().lower()
    if execution_backend in {"gcp", "cloud", "cloud_run", "cloud_run_jobs", "jobs"}:
        try:
            from models.session import _gcp_session_doc
            import json as _json
            _meta_path = os.path.join(session_path, "meta.json")
            _current_status = None
            try:
                _doc = _gcp_session_doc(session_id)
                _current_status = (_doc or {}).get("status")
            except Exception:
                try:
                    with open(_meta_path) as _f:
                        _current_status = _json.load(_f).get("status")
                except (FileNotFoundError, ValueError):
                    pass
            if _current_status == "MINING":
                logger.warning(
                    "mine_v2 GCP rejected: session %s already MINING", session_id
                )
                return jsonify({"error": "Mining already in progress", "status": "MINING"}), 409

            request_body = request.get_json(silent=True) or {}
            if request_body:
                _saved_config = load_session_config(session_path)
                config_dict = merge_configs(_saved_config.to_dict(), request_body)
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
                sync_file(session_id, "config.json")

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

    import json as _json
    _meta_path = os.path.join(session_path, "meta.json")
    try:
        with open(_meta_path) as _f:
            _current_status = _json.load(_f).get("status", "")
        if _current_status == "MINING":
            logger.warning("mine_v2 rejected: session %s already MINING", session_id)
            return jsonify({"error": "Mining already in progress", "status": "MINING"}), 409
    except (FileNotFoundError, ValueError):
        pass

    initialize_session_directories(session_path)

    import time as _time
    _mine_start = _time.monotonic()
    logger.info("mine_v2 starting: session=%s", session_id)

    # Step 1: Load data
    _t = _time.monotonic()
    logger.info("Step 1: Loading processed data")
    matrix, user_ids, ent_ids = load_sparse_matrix(session_id)
    identities = load_dataframe(session_id, "processed/identities.csv")

    catalog = None
    catalog_path = os.path.join(session_path, "processed", "catalog.csv")
    if os.path.isfile(catalog_path):
        catalog = load_dataframe(session_id, "processed/catalog.csv")
        logger.info("Loaded catalog: %d entitlements", len(catalog))

    logger.info(
        "Step 1 done: matrix=%d users x %d entitlements  elapsed_ms=%.0f",
        matrix.shape[0], matrix.shape[1], (_time.monotonic() - _t) * 1000,
    )

    # Step 2: Load saved config, then apply any request-body overrides on top.
    _t = _time.monotonic()
    logger.info("Step 2: Loading configuration")
    fetch_file(session_id, "config.json")
    config = load_session_config(session_path)
    request_body = request.get_json(silent=True) or {}
    if request_body:
        config_dict = merge_configs(config.to_dict(), request_body)
        config = MiningConfig.from_dict(config_dict)
        logger.info("Step 2: request-body overrides applied: %s", list(request_body.keys()))
    else:
        logger.info("Step 2: using saved config (no request-body overrides)")
    logger.info("Step 2 done: elapsed_ms=%.0f", (_time.monotonic() - _t) * 1000)

    # Step 3: Validate config
    _t = _time.monotonic()
    logger.info("Step 3: Validating configuration")
    validation_errors = config.validate()
    if validation_errors:
        return jsonify({
            "error": "Invalid configuration",
            "validation_errors": validation_errors,
        }), 400

    data_validation_errors = config.validate_against_data(
        identities_columns=list(identities.columns)
    )
    if data_validation_errors:
        return jsonify({
            "error": "Mining configuration error: attribute columns not found in data",
            "validation_errors": data_validation_errors,
        }), 400

    save_session_config(session_path, config)
    sync_file(session_id, "config.json")
    logger.info("Step 3 done: elapsed_ms=%.0f", (_time.monotonic() - _t) * 1000)

    # Flatten per-app birthright dict to namespaced list
    birthright_explicit_flat = []
    for app_name, ent_list in config.birthright_explicit.items():
        for eid in ent_list:
            birthright_explicit_flat.append(f"{app_name}:{eid}")

    # Step 4: Birthright detection (reuse V1)
    _t = _time.monotonic()
    logger.info("Step 4: Detecting birthright entitlements")
    birthright_result = detect_birthright(
        matrix=matrix,
        user_ids=user_ids,
        ent_ids=ent_ids,
        threshold=config.birthright_threshold,
        explicit_list=birthright_explicit_flat,
        min_assignment_count=config.min_assignment_count,
    )
    logger.info(
        "Step 4 done: birthright=%d noise=%d kept=%d  elapsed_ms=%.0f",
        len(birthright_result["birthright_entitlements"]),
        len(birthright_result.get("noise_entitlements", [])),
        len(birthright_result["filtered_ent_ids"]),
        (_time.monotonic() - _t) * 1000,
    )

    # Step 4.5: Tiered birthright detection (sub-population birthrights)
    tiered_result = None
    if config.tiered_birthright_enabled:
        _t = _time.monotonic()
        logger.info("Step 4.5: Tiered birthright detection")

        attr_columns = config.get_attribute_columns() if hasattr(config, 'get_attribute_columns') else []
        if not attr_columns:
            logger.info(
                "Step 4.5: skipping tiered birthright detection — "
                "user_attributes not configured. Configure via PUT /config to enable."
            )
        else:
            tiered_result = detect_tiered_birthrights(
                matrix=birthright_result["filtered_matrix"],
                user_ids=user_ids,
                ent_ids=birthright_result["filtered_ent_ids"],
                identities=identities,
                user_attributes=config.user_attributes,
                threshold=config.tiered_birthright_threshold,
                min_subpop_size=config.tiered_birthright_min_subpop_size,
            )
            logger.info(
                "Step 4.5 done: tiered_roles=%d pairs_absorbed=%d elapsed_ms=%.0f",
                len(tiered_result["tiered_birthright_roles"]),
                tiered_result["tiered_stats"]["total_assignment_pairs_absorbed"],
                (_time.monotonic() - _t) * 1000,
            )
    else:
        logger.info("Step 4.5: tiered birthright detection disabled")

    # Step 5: Entitlement clustering (V2 - Leiden)
    _t = _time.monotonic()
    logger.info("Step 5: Running Leiden clustering")

    clustering_matrix = (
        tiered_result["filtered_matrix"]
        if tiered_result
        else birthright_result["filtered_matrix"]
    )
    clustering_ent_ids = (
        tiered_result["filtered_ent_ids"]
        if tiered_result
        else birthright_result["filtered_ent_ids"]
    )

    try:
        cluster_result = cluster_entitlements_leiden(
            matrix=clustering_matrix,
            ent_ids=clustering_ent_ids,
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
        logger.error("Step 5 clustering failed: %s", e, exc_info=True)
        return jsonify({"error": "Clustering failed", "details": str(e)}), 500
    logger.info(
        "Step 5 done: clusters=%d cpm_quality=%.3f assigned_users=%d  elapsed_ms=%.0f",
        cluster_result["n_clusters"],
        cluster_result["leiden_stats"]["cpm_quality"],
        len(cluster_result["user_cluster_membership"]),
        (_time.monotonic() - _t) * 1000,
    )

    # Step 6: Build roles from clusters (V2)
    _t = _time.monotonic()
    logger.info("Step 6: Building roles from clusters")
    roles_result = build_roles(
        matrix=matrix,
        user_ids=user_ids,
        ent_ids=ent_ids,
        cluster_result=cluster_result,
        birthright_result=birthright_result,
        config=config.to_dict(),
        tiered_birthright_roles=(
            tiered_result["tiered_birthright_roles"] if tiered_result else []
        ),
    )
    logger.info(
        "Step 6 done: roles=%d residuals=%d  elapsed_ms=%.0f",
        len(roles_result.get("roles", [])),
        len(roles_result.get("residuals", [])),
        (_time.monotonic() - _t) * 1000,
    )

    draft_roles = roles_result["roles"]
    birthright_role = roles_result["birthright_role"]
    residuals = roles_result["residuals"]
    multi_cluster_info = roles_result["multi_cluster_info"]
    naming_summary = roles_result["naming_summary"]
    summary_base = roles_result["summary"]
    birthright_promotions = roles_result.get("birthright_promotions", [])
    merge_candidates = roles_result.get("merge_candidates", [])

    # ---- Step 6.5: Hybrid co-occurrence mining on residuals ----
    # CHANGE 2026-04-02: Discover supplementary roles from Leiden residuals
    _t = _time.monotonic()
    logger.info("Step 6.5: Hybrid co-occurrence mining on residuals")
    hybrid_result = discover_hybrid_roles(
        matrix=matrix,
        user_ids=user_ids,
        ent_ids=ent_ids,
        leiden_roles=draft_roles,
        user_memberships=cluster_result["user_cluster_membership"],
        birthright_entitlements=birthright_result["birthright_entitlements"],
        tiered_birthright_roles=(
            tiered_result["tiered_birthright_roles"] if tiered_result else []
        ),
        min_co_occurrence=config.leiden_min_shared_users * 8,  # scale with config
        min_co_occurrence_growth=getattr(config, 'min_co_occurrence_growth', 10),
        min_role_users=config.min_role_size,
        min_entitlements_per_role=config.min_entitlements_per_role,
    )
    hybrid_roles = hybrid_result["hybrid_roles"]
    hybrid_stats = hybrid_result["hybrid_stats"]

    if hybrid_roles:
        # Merge hybrid roles into cluster_result so confidence scorer can find them
        cluster_result = merge_hybrid_into_cluster_result(cluster_result, hybrid_roles)
        # Append hybrid roles to draft_roles list
        draft_roles = draft_roles + hybrid_roles
        logger.info(
            "Step 6.5 done: %d hybrid roles, %d users covered, elapsed_ms=%.0f",
            len(hybrid_roles), hybrid_stats["hybrid_users_covered"],
            (_time.monotonic() - _t) * 1000,
        )
    else:
        logger.info("Step 6.5 done: no hybrid roles discovered, elapsed_ms=%.0f",
                     (_time.monotonic() - _t) * 1000)

    # ---- Diagnostic: unassigned user entitlement distribution ----
    _t_diag = _time.monotonic()
    try:
        assigned_user_set = set(cluster_result["user_cluster_membership"].keys())
        user_id_to_idx = {uid: i for i, uid in enumerate(user_ids)}
        unassigned_indices = np.array([
            user_id_to_idx[uid] for uid in user_ids
            if str(uid) not in assigned_user_set and uid not in assigned_user_set
        ])
        if len(unassigned_indices) > 0:
            unassigned_sub = clustering_matrix[unassigned_indices]
            epts = np.diff(unassigned_sub.indptr)
            logger.info(
                "DIAGNOSTIC — unassigned user ent distribution (post-BR/tiered removal): "
                "n=%d mean=%.1f median=%.1f p25=%.0f p75=%.0f p90=%.0f max=%d "
                "zero=%d one_to_three=%d four_to_ten=%d ten_plus=%d",
                len(epts), epts.mean(), np.median(epts),
                np.percentile(epts, 25), np.percentile(epts, 75),
                np.percentile(epts, 90), epts.max(),
                int((epts == 0).sum()),
                int(((epts >= 1) & (epts <= 3)).sum()),
                int(((epts >= 4) & (epts <= 10)).sum()),
                int((epts > 10).sum()),
            )
        else:
            logger.info("DIAGNOSTIC — no unassigned users (all assigned to clusters)")
    except Exception as _diag_err:
        logger.warning("DIAGNOSTIC failed: %s", _diag_err, exc_info=True)
    logger.info("DIAGNOSTIC elapsed_ms=%.0f", (_time.monotonic() - _t_diag) * 1000)

    # Update summary with hybrid stats
    summary_base["hybrid_roles"] = len(hybrid_roles)
    summary_base["hybrid_users_covered"] = hybrid_stats.get("hybrid_users_covered", 0)
    summary_base["total_roles"] = len(draft_roles)
    summary_base["assigned_users"] = len(cluster_result["user_cluster_membership"])
    summary_base["unassigned_users"] = matrix.shape[0] - summary_base["assigned_users"]

    # Step 7: Build results structure
    results = {
        "roles": draft_roles,
        "birthright_role": birthright_role,
        "tiered_birthright_roles": (
            tiered_result["tiered_birthright_roles"] if tiered_result else []
        ),
        "tiered_birthright_stats": (
            tiered_result["tiered_stats"] if tiered_result else {}
        ),
        "hybrid_stats": hybrid_stats,
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
        "status": "draft",
    }

    # Step 8: Confidence Scoring (V2 - Multi-factor)
    _t = _time.monotonic()
    logger.info("Step 8: Computing multi-factor confidence scores")
    _scoring_config_errors = config.validate_for_confidence_scoring()
    if _scoring_config_errors:
        logger.warning(
            "Step 8: skipping confidence scoring — user_attributes not configured: %s",
            _scoring_config_errors,
        )
        results["confidence_scoring"] = {
            "skipped": True,
            "reason": _scoring_config_errors,
            "message": "Configure user_attributes via PUT /config to enable confidence scoring",
        }
    else:
        assignments_path = os.path.join(session_path, "processed", "assignments.csv")
        if not os.path.isfile(assignments_path):
            logger.warning("Step 8: assignments.csv not found, skipping confidence scoring")
        elif not CONFIDENCE_SCORER_V2_AVAILABLE:
            logger.warning("Step 8: confidence_scorer not installed, skipping")
            results["confidence_scoring"] = {
                "skipped": True,
                "reason": "confidence_scorer module not installed",
            }

    if os.path.isfile(os.path.join(session_path, "processed", "assignments.csv")) \
            and CONFIDENCE_SCORER_V2_AVAILABLE \
            and not _scoring_config_errors:
        assignments_path = os.path.join(session_path, "processed", "assignments.csv")
        assignments_df = load_dataframe(session_id, "processed/assignments.csv").reset_index(drop=True)

        if CONFIDENCE_SCORER_V2_AVAILABLE:
            try:
                from services.confidence_scorer import SparseMatrixAccessor
                matrix_for_scoring = SparseMatrixAccessor(matrix, user_ids, ent_ids)

                enriched_assignments = score_assignments(
                    assignments_df=assignments_df,
                    full_matrix=matrix_for_scoring,
                    identities=identities,
                    cluster_result=cluster_result,
                    roles=draft_roles,
                    birthright_entitlements=birthright_result["birthright_entitlements"],
                    noise_entitlements=birthright_result.get("noise_entitlements", []),
                    tiered_birthright_roles=(
                        tiered_result["tiered_birthright_roles"] if tiered_result else []
                    ),
                    config=config.to_dict(),
                    drift_data=None,
                )

                results_path = get_results_path(session_path)
                enriched_assignments.to_csv(
                    os.path.join(results_path, "assignments_scored.csv"),
                    index=False
                )

                recommendations = generate_recommendations(
                    enriched_assignments=enriched_assignments,
                    full_matrix=matrix_for_scoring,
                    cluster_result=cluster_result,
                    config=config.to_dict(),
                )

                over_provisioned = detect_over_provisioned(
                    enriched_assignments=enriched_assignments,
                    revocation_threshold=config.revocation_threshold,
                )

                recommendations.to_csv(
                    os.path.join(results_path, "recommendations.csv"),
                    index=False
                )
                over_provisioned.to_csv(
                    os.path.join(results_path, "over_provisioned.csv"),
                    index=False
                )

                scoring_summary = build_scoring_summary(
                    enriched_assignments=enriched_assignments,
                    cluster_result=cluster_result,
                    birthright_entitlements=birthright_result["birthright_entitlements"],
                )

                results.update(scoring_summary)

                logger.info(
                    f"Confidence scoring complete: "
                    f"{scoring_summary['confidence_scoring']['high']} HIGH, "
                    f"{scoring_summary['confidence_scoring']['medium']} MEDIUM, "
                    f"{scoring_summary['confidence_scoring']['low']} LOW"
                )

            except Exception as e:
                logger.error(f"Confidence scoring failed: {e}", exc_info=True)
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

    # Step 9: Save results
    _t = _time.monotonic()
    logger.info("Step 9: Saving results")

    results = _sanitize_for_json(results)

    save_json(session_id, "results/draft_results.json", results)

    try:
        logger.info("Step 9: Patching stats.json")
        fetch_file(session_id, "processed/stats.json")
    except Exception:
        pass
    _stats_path = os.path.join(session_path, "processed", "stats.json")
    if os.path.isfile(_stats_path):
        try:
            with open(_stats_path) as _f:
                _stats = json.load(_f)
            _stats["roles_discovered"] = summary_base.get("total_roles", 0)
            with open(_stats_path, "w") as _f:
                json.dump(_stats, _f, indent=2)
            sync_file(session_id, "processed/stats.json")
        except Exception as _e:
            logger.warning("Failed to patch stats.json with roles_discovered: %s", _e, exc_info=True)

    save_json(
        session_id,
        "results/cluster_membership.json",
        cluster_result["user_cluster_membership"],
    )
    update_session_fields(session_id, {"status": "MINED"})
    sync_session_tree(session_id)
    logger.info(
        "Step 9 done: elapsed_ms=%.0f", (_time.monotonic() - _t) * 1000
    )
    logger.info(
        "mine_v2 complete: session=%s total_elapsed_ms=%.0f roles=%d (leiden=%d hybrid=%d)",
        session_id,
        (_time.monotonic() - _mine_start) * 1000,
        len(results.get("roles", [])),
        len(roles_result.get("roles", [])),
        len(hybrid_roles),
    )
    return jsonify(results), 200


# ============================================================================
# DRAFT ROLE MANAGEMENT
# ============================================================================

@mining_bp.route("/api/sessions/<session_id>/draft-roles", methods=["GET"])

@require_auth
def get_draft_roles(session_id):
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

    role = None
    for r in draft_results["roles"]:
        if r["role_id"] == role_id:
            role = r
            break

    if not role:
        return jsonify({"error": f"Role {role_id} not found"}), 404

    body = request.get_json(silent=True) or {}

    if "owner" not in body:
        return jsonify({"error": "Field 'owner' is required"}), 400

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

def _build_summary(cluster_result: dict, birthright_result: dict, matrix) -> dict:
    """Build summary statistics."""
    n_clusters = cluster_result["n_clusters"]
    user_memberships = cluster_result["user_cluster_membership"]

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