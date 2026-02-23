#!/usr/bin/env python3
"""
Cloud Run Jobs worker entrypoint.

This is the "main program" that Cloud Run Jobs executes inside the container.

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

import os
import sys
import traceback
from typing import Any, Dict


def _require_env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return v


def _set_status(session_id: str, status: str, extra: Dict[str, Any] | None = None) -> None:
    # Imported lazily so worker can at least fail gracefully if imports break.
    from models.session import update_session_fields

    payload: Dict[str, Any] = {"status": status}
    if extra:
        payload.update(extra)
    update_session_fields(session_id, payload)


def _run_process(session_id: str) -> Dict[str, Any]:
    from models.session import get_session_path, save_json, sync_session_tree
    from services.data_loader import process_upload

    session_path = get_session_path(session_id)

    _set_status(session_id, "PROCESSING")

    stats = process_upload(session_path)
    save_json(session_id, "processed/stats.json", stats)

    # Persist any artifacts written by process_upload() (matrix.npz, csvs, etc.)
    sync_session_tree(session_id)

    _set_status(session_id, "PROCESSED")
    return {"stats": stats}


def _run_mine(session_id: str) -> Dict[str, Any]:
    """
    Runs the same pipeline as POST /api/sessions/<id>/mine (V2),
    but config comes from config.json on disk (or defaults).
    """
    import os

    from models.session import (
        get_session_path,
        save_json,
        load_dataframe,
        sync_session_tree,
        update_session_fields,
    )

    # Reuse the same module the Flask routes use so behavior stays aligned.
    from routes import mining as mining_routes

    import time as _time

    _t0 = _time.monotonic()
    def _elapsed():
        return (_time.monotonic() - _t0) * 1000

    print("Step 1: Loading session data", flush=True)
    _t = _time.monotonic()
    session_path = get_session_path(session_id)

    # Basic precondition: must have processed matrix
    matrix_path = os.path.join(session_path, "processed", "matrix.npz")
    if not os.path.isfile(matrix_path):
        raise RuntimeError("No processed data. Call /process first.")

    update_session_fields(session_id, {"status": "MINING"})

    # Create expected output directories
    mining_routes.initialize_session_directories(session_path)

    matrix, user_ids, ent_ids = mining_routes.load_sparse_matrix(session_id)
    identities = load_dataframe(session_id, "processed/identities.csv")
    print(f"Step 1 done: elapsed_ms={(_time.monotonic()-_t)*1000:.0f} users={len(user_ids)} ents={len(ent_ids)}", flush=True)

    print("Step 2: Loading config", flush=True)
    _t = _time.monotonic()
    config = mining_routes.load_session_config(session_path)
    errors = config.validate()
    if errors:
        raise RuntimeError(f"Invalid configuration: {errors}")
    data_errors = config.validate_against_data(identities_columns=list(identities.columns))
    if data_errors:
        raise RuntimeError(f"Mining configuration error: {data_errors}")
    mining_routes.save_session_config(session_path, config)
    print(f"Step 2 done: elapsed_ms={(_time.monotonic()-_t)*1000:.0f}", flush=True)

    print("Step 3: Birthright detection", flush=True)
    _t = _time.monotonic()
    birthright_explicit_flat = []
    try:
        for app_name, ent_list in (config.birthright_explicit or {}).items():
            for eid in ent_list:
                birthright_explicit_flat.append(f"{app_name}:{eid}")
    except Exception:
        birthright_explicit_flat = []

    birthright_result = mining_routes.detect_birthright(
        matrix=matrix,
        user_ids=user_ids,
        ent_ids=ent_ids,
        threshold=config.birthright_threshold,
        explicit_list=birthright_explicit_flat,
        min_assignment_count=config.min_assignment_count,
    )
    print(f"Step 3 done: elapsed_ms={(_time.monotonic()-_t)*1000:.0f} birthright_ents={len(birthright_result['birthright_entitlements'])}", flush=True)

    print("Step 4: Leiden clustering", flush=True)
    _t = _time.monotonic()
    cluster_result = mining_routes.cluster_entitlements_leiden(
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
    print(f"Step 4 done: elapsed_ms={(_time.monotonic()-_t)*1000:.0f} clusters={cluster_result['n_clusters']}", flush=True)

    print("Step 5: Building roles", flush=True)
    _t = _time.monotonic()
    roles_result = mining_routes.build_roles(
        matrix=matrix,
        user_ids=user_ids,
        ent_ids=ent_ids,
        cluster_result=cluster_result,
        birthright_result=birthright_result,
        config=config.to_dict(),
    )
    print(f"Step 5 done: elapsed_ms={(_time.monotonic()-_t)*1000:.0f} roles={len(roles_result['roles'])}", flush=True)

    from config.config import get_results_path

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

    # Step 8: Confidence Scoring (mirrors routes/mining.py)
    try:
        from services.confidence_scorer import (
            score_assignments,
            generate_recommendations,
            detect_over_provisioned,
            build_scoring_summary,
            SparseMatrixAccessor,
        )
        confidence_scorer_available = True
    except ImportError:
        confidence_scorer_available = False

    _scoring_config_errors = config.validate_for_confidence_scoring()
    assignments_path = os.path.join(session_path, "processed", "assignments.csv")

    if _scoring_config_errors:
        print(f"Step 8: skipping confidence scoring — user_attributes not configured: {_scoring_config_errors}", flush=True)
        results["confidence_scoring"] = {
            "skipped": True,
            "reason": _scoring_config_errors,
            "message": "Configure user_attributes via PUT /config to enable confidence scoring",
        }
    elif not os.path.isfile(assignments_path):
        print("Step 8: assignments.csv not found, skipping confidence scoring", flush=True)
    elif not confidence_scorer_available:
        print("Step 8: confidence_scorer not installed, skipping", flush=True)
        results["confidence_scoring"] = {
            "skipped": True,
            "reason": "confidence_scorer module not installed",
        }
    else:
        try:
            print("Step 8: Computing multi-factor confidence scores", flush=True)
            assignments_df = load_dataframe(session_id, "processed/assignments.csv").reset_index(drop=True)
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
                os.path.join(results_path, "assignments_scored.csv"),
                index=False
            )

            recommendations = generate_recommendations(
                enriched_assignments=enriched_assignments,
                full_matrix=matrix_for_scoring,
                cluster_result=cluster_result,
                config=config.to_dict(),
            )
            recommendations.to_csv(
                os.path.join(results_path, "recommendations.csv"),
                index=False
            )

            over_provisioned = detect_over_provisioned(
                enriched_assignments=enriched_assignments,
                revocation_threshold=config.revocation_threshold,
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

            print(
                f"Step 8 done: {scoring_summary['confidence_scoring']['high']} HIGH, "
                f"{scoring_summary['confidence_scoring']['medium']} MEDIUM, "
                f"{scoring_summary['confidence_scoring']['low']} LOW",
                flush=True
            )

        except Exception as e:
            print(f"Step 8: confidence scoring failed: {e}", flush=True)
            traceback.print_exc()
            results["confidence_scoring"] = {
                "error": str(e),
                "message": "Confidence scoring failed, results available without scoring",
            }

    # Step 9: Save results
    print("Step 9: Saving results", flush=True)
    save_json(session_id, "results/draft_results.json", results)
    if "user_cluster_membership" in cluster_result:
        save_json(session_id, "results/cluster_membership.json", cluster_result["user_cluster_membership"])

    sync_session_tree(session_id)
    update_session_fields(session_id, {"status": "MINED"})
    print(f"Mining complete: {len(results.get('roles', []))} roles total_elapsed_ms={_elapsed():.0f}", flush=True)
    return results


def main() -> int:
    try:
        mode = (sys.argv[1] if len(sys.argv) > 1 else "").strip().lower()
        if mode not in {"process", "mine"}:
            raise RuntimeError("Usage: python worker.py process|mine")

        session_id = _require_env("SESSION_ID")

        if mode == "process":
            _run_process(session_id)
        else:
            _run_mine(session_id)

        return 0

    except Exception as e:
        # Best-effort: mark session FAILED
        try:
            session_id = os.getenv("SESSION_ID")
            if session_id:
                _set_status(session_id, "FAILED", {"error": str(e)})
        except Exception:
            pass

        # Log to stderr for Cloud Run logs
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())