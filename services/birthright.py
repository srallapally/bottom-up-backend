# services/birthright.py
# CHANGE 2026-02-17: Added scipy.sparse import for sparse matrix support
import pandas as pd
import numpy as np
import time
from scipy.sparse import csr_matrix


def detect_birthright(
        matrix,  # CHANGE 2026-02-17: Now accepts csr_matrix + indices
        user_ids,
        ent_ids,
        threshold: float,
        explicit_list: list[str],
        min_assignment_count: int,
) -> dict:
    """
    Identifies birthright entitlements and filters the matrix.

    CHANGE 2026-02-17: Updated to work with sparse matrix instead of DataFrame.

    Args:
        matrix: scipy.sparse.csr_matrix (users × entitlements)
        user_ids: array-like of user IDs (row labels)
        ent_ids: array-like of entitlement IDs (column labels)
        threshold: birthright threshold (e.g., 0.8 = 80%)
        explicit_list: explicit birthright entitlement IDs
        min_assignment_count: minimum assignments to not be considered noise

    Returns dict with:
        - birthright_entitlements: list of entitlement IDs detected as birthright
        - noise_entitlements: list of entitlement IDs below min_assignment_count
        - filtered_matrix: sparse matrix with birthright and noise columns removed
        - filtered_ent_ids: entitlement IDs for filtered matrix
        - birthright_stats: per-entitlement assignment percentage
    """
    # FIX 19: track total elapsed for diagnostics
    _t_start = time.monotonic()

    # CHANGE 2026-02-17: Use sparse operations
    col_counts = np.array(matrix.sum(axis=0)).flatten()  # Convert to 1D array
    total_users = matrix.shape[0]
    col_pct = col_counts / total_users

    # Build mask as boolean array
    ent_ids_array = np.array(ent_ids)

    # Birthright: above threshold OR in explicit list
    birthright_mask = col_pct >= threshold
    for ent_id in explicit_list:
        idx = np.where(ent_ids_array == ent_id)[0]
        if len(idx) > 0:
            birthright_mask[idx[0]] = True

    birthright_entitlements = ent_ids_array[birthright_mask].tolist()

    # Noise: below min assignment count (excluding already-flagged birthright)
    noise_mask = (col_counts < min_assignment_count) & ~birthright_mask
    noise_entitlements = ent_ids_array[noise_mask].tolist()

    # Filter: keep columns that are neither birthright nor noise
    keep_mask = ~birthright_mask & ~noise_mask
    keep_indices = np.where(keep_mask)[0]

    # CHANGE 2026-02-17: Column slicing on sparse matrix
    filtered_matrix = matrix[:, keep_indices]
    filtered_ent_ids = ent_ids_array[keep_indices].tolist()

    # Stats for birthright
    birthright_stats = {}
    for i, ent in enumerate(birthright_entitlements):
        idx = np.where(ent_ids_array == ent)[0][0]
        birthright_stats[ent] = {
            "count": int(col_counts[idx]),
            "pct": round(float(col_pct[idx]), 4)
        }

    import logging as _logging
    _logging.getLogger(__name__).info(
        "birthright detection complete: total=%d birthright=%d noise=%d kept=%d elapsed_ms=%.0f",
        len(ent_ids),
        len(birthright_entitlements),
        len(noise_entitlements),
        len(filtered_ent_ids),
        (time.monotonic() - _t_start) * 1000,
    )
    return {
        "birthright_entitlements": birthright_entitlements,
        "noise_entitlements": noise_entitlements,
        "filtered_matrix": filtered_matrix,
        "filtered_ent_ids": filtered_ent_ids,  # CHANGE 2026-02-17: Return filtered column labels
        "birthright_stats": birthright_stats,
    }