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

    session_path = get_session_path(session_id)

    # Basic precondition: must have processed matrix
    matrix_path = os.path.join(session_path, "processed", "matrix.npz")
    if not os.path.isfile(matrix_path):
        raise RuntimeError("No processed data. Call /process first.")

    update_session_fields(session_id, {"status": "MINING"})

    # Create expected output directories
    mining_routes.initialize_session_directories(session_path)

    # Load processed data
    matrix, user_ids, ent_ids = mining_routes.load_sparse_matrix(session_id)
    identities = load_dataframe(session_id, "processed/identities.csv")

    catalog = None
    catalog_path = os.path.join(session_path, "processed", "catalog.csv")
    if os.path.isfile(catalog_path):
        catalog = load_dataframe(session_id, "processed/catalog.csv")

    # Load config from session (config.json) or defaults
    config = mining_routes.load_session_config(session_path)

    # Validate config
    errors = config.validate()
    if errors:
        raise RuntimeError(f"Invalid configuration: {errors}")

    data_errors = config.validate_against_data(identities_columns=list(identities.columns))
    if data_errors:
        raise RuntimeError(f"Mining configuration error: {data_errors}")

    # Save config for reproducibility (no-op if already saved)
    mining_routes.save_session_config(session_path, config)

    # Flatten explicit birthright list: {app:[ent,...]} -> ["app:ent", ...]
    birthright_explicit_flat = []
    try:
        for app_name, ent_list in (config.birthright_explicit or {}).items():
            for eid in ent_list:
                birthright_explicit_flat.append(f"{app_name}:{eid}")
    except Exception:
        # Keep it robust if config shape differs
        birthright_explicit_flat = []

    # Step: birthright detection (mirrors routes/mining.py)
    birthright_result = mining_routes.detect_birthright(
        matrix=matrix,
        user_ids=user_ids,
        ent_ids=ent_ids,
        threshold=config.birthright_threshold,
        explicit_list=birthright_explicit_flat,
        min_assignment_count=config.min_assignment_count,
    )

    # Step: Leiden clustering (mirrors routes/mining.py)
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

    # Step: build roles (mirrors routes/mining.py)
    roles_result = mining_routes.build_roles(
        matrix=matrix,
        user_ids=user_ids,
        ent_ids=ent_ids,
        cluster_result=cluster_result,
        birthright_result=birthright_result,
        identities=identities,
        catalog=catalog,
        config=config.to_dict(),
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

    # Save draft results (the UI reads this)
    save_json(session_id, "results/draft_results.json", results)
    # Persist cluster membership if present
    if "user_cluster_membership" in cluster_result:
        save_json(session_id, "results/cluster_membership.json", cluster_result["user_cluster_membership"])

    # Persist all written artifacts to GCS (in gcp mode)
    sync_session_tree(session_id)

    update_session_fields(session_id, {"status": "MINED"})
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