#!/usr/bin/env python3
# worker.py
"""
Cloud Run Jobs worker entrypoint.

Usage:
  python worker.py process
  python worker.py mine

Required env:
  SESSION_ID=<session_id>

Behavior:
- Uses models.session.get_session_path(session_id) which supports dual store:
    * local: returns BASE_DATA_DIR/<session_id>
    * gcp: downloads gs://.../users/<sub>/sessions/<id>/ into /tmp/... and returns that path
- Runs the same processing/mining logic used by the Flask routes,
  but without Flask request/response objects.
- Syncs outputs back to GCS in gcp mode via sync_session_tree().
- Updates session status (local meta.json or Firestore depending on backend).
"""

import json
import logging
import os
import sys
import time
import traceback
from typing import Any, Dict

# FIX 18: module-level logger so Cloud Run captures structured output on stderr.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("worker")


def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v


def _set_status(
    session_id: str,
    status: str,
    extra: Dict[str, Any] | None = None,
    old_status: str | None = None,
) -> None:
    """Update session status with transition logging (Fix 22)."""
    from models.session import update_session_fields

    payload: Dict[str, Any] = {"status": status}
    if extra:
        payload.update(extra)
    update_session_fields(session_id, payload)
    # FIX 22: log every transition so lifecycle is traceable in Cloud Logging.
    logger.info(
        "status_transition session=%s %s-> %s",
        session_id,
        f"{old_status} " if old_status else "",
        status,
    )


def _read_current_status(session_path: str) -> str | None:
    """
    Read current session status from meta.json.

    update_session_fields() always writes meta.json on both local and GCP backends,
    so this is authoritative without needing a Firestore round-trip.
    """
    meta_path = os.path.join(session_path, "meta.json")
    try:
        with open(meta_path) as f:
            return json.load(f).get("status")
    except (FileNotFoundError, ValueError):
        return None


# ============================================================================
# PROCESS WORKER
# ============================================================================

def _run_process(session_id: str) -> Dict[str, Any]:
    """
    Equivalent to POST /api/sessions/<id>/process, run synchronously.
    Fix 10: re-entrancy guard.
    Fix 18: per-step timing.
    Fix 22: status transition logging.
    """
    from models.session import get_session_path, save_json, sync_session_tree
    from services.data_loader import process_upload

    t_start = time.monotonic()
    logger.info("_run_process starting: session=%s", session_id)

    session_path = get_session_path(session_id)

    # FIX 10: guard against duplicate process jobs.
    current = _read_current_status(session_path)
    if current == "PROCESSING":
        raise RuntimeError(
            f"Session {session_id} is already PROCESSING — duplicate job, aborting."
        )

    _set_status(session_id, "PROCESSING", old_status=current)

    # Step 1: parse uploads, build matrix
    t = time.monotonic()
    logger.info("_run_process step=1 name=process_upload")
    stats = process_upload(session_path)
    logger.info(
        "_run_process step=1 done: users=%d entitlements=%d assignments=%d elapsed_ms=%.0f",
        stats.get("total_users", 0),
        stats.get("total_entitlements", 0),
        stats.get("total_assignments", 0),
        (time.monotonic() - t) * 1000,
    )

    # Step 2: save stats
    t = time.monotonic()
    save_json(session_id, "processed/stats.json", stats)
    logger.info("_run_process step=2 name=save_stats elapsed_ms=%.0f", (time.monotonic() - t) * 1000)

    # Step 3: sync artifacts to GCS
    t = time.monotonic()
    logger.info("_run_process step=3 name=sync_to_gcs")
    sync_session_tree(session_id)
    logger.info("_run_process step=3 done elapsed_ms=%.0f", (time.monotonic() - t) * 1000)

    _set_status(session_id, "PROCESSED", old_status="PROCESSING")
    logger.info(
        "_run_process complete: session=%s total_elapsed_ms=%.0f",
        session_id, (time.monotonic() - t_start) * 1000,
    )
    return {"stats": stats}


# ============================================================================
# MINE WORKER
# ============================================================================

def _run_mine(session_id: str) -> Dict[str, Any]:
    """
    Equivalent to POST /api/sessions/<id>/mine, run synchronously.

    Fix 10: re-entrancy guard — raises if session is already MINING so a
            duplicate Cloud Run Job exits immediately without corrupting results.
    Fix 11: imports directly from services.* and config.* instead of routing
            through routes.mining (fragile outside Flask context).
    Fix 18: per-step timing logged to stderr → Cloud Logging.
    Fix 22: status transitions logged with old→new.
    Fix 3 wire-in: confidence scoring gated on validate_for_confidence_scoring().
    """
    # FIX 11: Direct imports — no dependency on routes.mining.
    from models.session import (
        get_session_path,
        load_dataframe,
        save_json,
        sync_session_tree,
    )
    from config.config import (
        initialize_session_directories,
        load_session_config,
        save_session_config,
        get_results_path,
    )
    from services.birthright import detect_birthright
    from services.clustering import cluster_entitlements_leiden
    from services.role_builder import build_roles
    import scipy.sparse as sp
    import pandas as pd

    def _load_sparse_matrix(session_path: str):
        matrix = sp.load_npz(os.path.join(session_path, "processed", "matrix.npz"))
        user_ids = pd.read_csv(
            os.path.join(session_path, "processed", "matrix_users.csv")
        )["USR_ID"].values
        ent_ids = pd.read_csv(
            os.path.join(session_path, "processed", "matrix_entitlements.csv")
        )["namespaced_id"].values
        return matrix, user_ids, ent_ids

    t_start = time.monotonic()
    logger.info("_run_mine starting: session=%s", session_id)

    session_path = get_session_path(session_id)

    # Precondition: processed matrix must exist.
    if not os.path.isfile(os.path.join(session_path, "processed", "matrix.npz")):
        raise RuntimeError("No processed data. Run the process job first.")

    # FIX 10: Re-entrancy guard. update_session_fields() always writes meta.json
    # on both backends, so this is authoritative without a Firestore round-trip.
    current_status = _read_current_status(session_path)
    if current_status == "MINING":
        raise RuntimeError(
            f"Session {session_id} is already MINING — duplicate job, aborting."
        )

    _set_status(session_id, "MINING", old_status=current_status)
    initialize_session_directories(session_path)

    # Step 1: Load data
    t = time.monotonic()
    logger.info("_run_mine step=1 name=load_data")
    matrix, user_ids, ent_ids = _load_sparse_matrix(session_path)
    identities = load_dataframe(session_id, "processed/identities.csv")
    catalog = None
    catalog_path = os.path.join(session_path, "processed", "catalog.csv")
    if os.path.isfile(catalog_path):
        catalog = load_dataframe(session_id, "processed/catalog.csv")
        logger.info("_run_mine step=1 catalog loaded: rows=%d", len(catalog))
    logger.info(
        "_run_mine step=1 done: matrix=%d users x %d entitlements elapsed_ms=%.0f",
        matrix.shape[0], matrix.shape[1], (time.monotonic() - t) * 1000,
    )

    # Step 2: Load config
    t = time.monotonic()
    logger.info("_run_mine step=2 name=load_config")
    config = load_session_config(session_path)
    logger.info("_run_mine step=2 done: elapsed_ms=%.0f", (time.monotonic() - t) * 1000)

    # Step 3: Validate config
    t = time.monotonic()
    logger.info("_run_mine step=3 name=validate_config")
    errors = config.validate()
    if errors:
        raise RuntimeError(f"Invalid configuration: {errors}")
    data_errors = config.validate_against_data(identities_columns=list(identities.columns))
    if data_errors:
        raise RuntimeError(f"Mining configuration error: {data_errors}")
    save_session_config(session_path, config)
    logger.info("_run_mine step=3 done: elapsed_ms=%.0f", (time.monotonic() - t) * 1000)

    # Flatten explicit birthright list: {app:[ent,...]} -> ["app:ent", ...]
    birthright_explicit_flat = []
    for app_name, ent_list in (config.birthright_explicit or {}).items():
        for eid in ent_list:
            birthright_explicit_flat.append(f"{app_name}:{eid}")

    # Step 4: Birthright detection
    t = time.monotonic()
    logger.info("_run_mine step=4 name=birthright_detection")
    birthright_result = detect_birthright(
        matrix=matrix,
        user_ids=user_ids,
        ent_ids=ent_ids,
        threshold=config.birthright_threshold,
        explicit_list=birthright_explicit_flat,
        min_assignment_count=config.min_assignment_count,
    )
    logger.info(
        "_run_mine step=4 done: birthright=%d noise=%d kept=%d elapsed_ms=%.0f",
        len(birthright_result["birthright_entitlements"]),
        len(birthright_result.get("noise_entitlements", [])),
        len(birthright_result["filtered_ent_ids"]),
        (time.monotonic() - t) * 1000,
    )

    # Step 5: Leiden clustering
    t = time.monotonic()
    logger.info("_run_mine step=5 name=leiden_clustering")
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
    logger.info(
        "_run_mine step=5 done: clusters=%d cpm_quality=%.3f assigned_users=%d elapsed_ms=%.0f",
        cluster_result["n_clusters"],
        cluster_result["leiden_stats"]["cpm_quality"],
        len(cluster_result["user_cluster_membership"]),
        (time.monotonic() - t) * 1000,
    )

    # Step 6: Build roles
    t = time.monotonic()
    logger.info("_run_mine step=6 name=build_roles")
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
    logger.info(
        "_run_mine step=6 done: roles=%d residuals=%d elapsed_ms=%.0f",
        len(roles_result.get("roles", [])),
        len(roles_result.get("residuals", [])),
        (time.monotonic() - t) * 1000,
    )

    results = {
        "roles": roles_result["roles"],
        "birthright_role": roles_result["birthright_role"],
        "birthright_promotions": roles_result.get("birthright_promotions", []),
        "merge_candidates": roles_result.get("merge_candidates", []),
        "residuals": roles_result["residuals"],
        "cluster_info": {
            "leiden_stats": cluster_result["leiden_stats"],
            "n_clusters": cluster_result["n_clusters"],
            "unassigned_users": cluster_result["unassigned_users"],
        },
        "summary": roles_result["summary"],
        "multi_cluster_info": roles_result["multi_cluster_info"],
        "naming_summary": roles_result["naming_summary"],
        "config": config.to_dict(),
        "status": "draft",
    }

    # Step 7: Confidence scoring
    # FIX 3 wire-in: gated on validate_for_confidence_scoring() so the worker
    # never fails when user_attributes are not yet configured.
    t = time.monotonic()
    logger.info("_run_mine step=7 name=confidence_scoring")
    _scoring_config_errors = config.validate_for_confidence_scoring()
    if _scoring_config_errors:
        logger.warning(
            "_run_mine step=7 skipped: user_attributes not configured: %s",
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
            logger.warning("_run_mine step=7 skipped: assignments.csv not found")
        else:
            try:
                from services.confidence_scorer import (
                    SparseMatrixAccessor,
                    score_assignments,
                    generate_recommendations,
                    detect_over_provisioned,
                    build_scoring_summary,
                )

                assignments_df = load_dataframe(
                    session_id, "processed/assignments.csv"
                ).reset_index(drop=True)

                matrix_for_scoring = SparseMatrixAccessor(matrix, user_ids, ent_ids)

                enriched_assignments = score_assignments(
                    assignments_df=assignments_df,
                    full_matrix=matrix_for_scoring,
                    identities=identities,
                    cluster_result=cluster_result,
                    roles=roles_result["roles"],
                    birthright_entitlements=birthright_result["birthright_entitlements"],
                    noise_entitlements=birthright_result.get("noise_entitlements", []),
                    config=config.to_dict(),
                    drift_data=None,
                )

                results_path = get_results_path(session_path)
                enriched_assignments.to_csv(
                    os.path.join(results_path, "assignments_scored.csv"), index=False
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
                    os.path.join(results_path, "recommendations.csv"), index=False
                )
                over_provisioned.to_csv(
                    os.path.join(results_path, "over_provisioned.csv"), index=False
                )

                scoring_summary = build_scoring_summary(
                    enriched_assignments=enriched_assignments,
                    cluster_result=cluster_result,
                    birthright_entitlements=birthright_result["birthright_entitlements"],
                )
                results.update(scoring_summary)

                logger.info(
                    "_run_mine step=7 done: high=%d medium=%d low=%d elapsed_ms=%.0f",
                    scoring_summary.get("confidence_scoring", {}).get("high", 0),
                    scoring_summary.get("confidence_scoring", {}).get("medium", 0),
                    scoring_summary.get("confidence_scoring", {}).get("low", 0),
                    (time.monotonic() - t) * 1000,
                )

            except ImportError:
                logger.warning("_run_mine step=7 skipped: confidence_scorer module not installed")
                results["confidence_scoring"] = {"skipped": True, "reason": "module not installed"}
            except Exception as exc:
                logger.error("_run_mine step=7 failed: %s", exc, exc_info=True)
                results["confidence_scoring"] = {
                    "error": str(exc),
                    "message": "Confidence scoring failed; results saved without scores",
                }

    # Step 8: Save and sync
    t = time.monotonic()
    logger.info("_run_mine step=8 name=save_and_sync")
    save_json(session_id, "results/draft_results.json", results)
    save_json(
        session_id,
        "results/cluster_membership.json",
        cluster_result["user_cluster_membership"],
    )
    sync_session_tree(session_id)
    logger.info("_run_mine step=8 done: elapsed_ms=%.0f", (time.monotonic() - t) * 1000)

    _set_status(session_id, "MINED", old_status="MINING")
    logger.info(
        "_run_mine complete: session=%s total_elapsed_ms=%.0f roles=%d",
        session_id,
        (time.monotonic() - t_start) * 1000,
        len(results.get("roles", [])),
    )
    return results


# ============================================================================
# ENTRYPOINT
# ============================================================================

def main() -> int:
    try:
        mode = (sys.argv[1] if len(sys.argv) > 1 else "").strip().lower()
        if mode not in {"process", "mine"}:
            raise RuntimeError("Usage: python worker.py process|mine")

        session_id = _require_env("SESSION_ID")
        logger.info("worker starting: mode=%s session=%s", mode, session_id)

        if mode == "process":
            _run_process(session_id)
        else:
            _run_mine(session_id)

        logger.info("worker finished: mode=%s session=%s", mode, session_id)
        return 0

    except Exception as e:
        try:
            session_id = os.getenv("SESSION_ID")
            if session_id:
                _set_status(session_id, "FAILED", extra={"error": str(e)})
        except Exception:
            pass
        traceback.print_exc()
        logger.error("worker failed: %s", e)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())