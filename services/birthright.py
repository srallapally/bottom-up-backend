import pandas as pd
import numpy as np


def detect_birthright(
    matrix: pd.DataFrame,
    threshold: float,
    explicit_list: list[str],
    min_assignment_count: int,
) -> dict:
    """
    Identifies birthright entitlements and filters the matrix.

    Returns dict with:
        - birthright_entitlements: list of entitlement IDs detected as birthright
        - noise_entitlements: list of entitlement IDs below min_assignment_count
        - filtered_matrix: matrix with birthright and noise columns removed
        - birthright_stats: per-entitlement assignment percentage
    """
    col_counts = matrix.sum(axis=0)
    total_users = len(matrix)
    col_pct = col_counts / total_users

    # Birthright: above threshold OR in explicit list
    birthright_mask = col_pct >= threshold
    for ent_id in explicit_list:
        if ent_id in matrix.columns:
            birthright_mask[ent_id] = True

    birthright_entitlements = matrix.columns[birthright_mask].tolist()

    # Noise: below min assignment count (excluding already-flagged birthright)
    noise_mask = (col_counts < min_assignment_count) & ~birthright_mask
    noise_entitlements = matrix.columns[noise_mask].tolist()

    # Filter
    keep_mask = ~birthright_mask & ~noise_mask
    filtered_matrix = matrix.loc[:, keep_mask]

    # Stats for birthright
    birthright_stats = {
        ent: {"count": int(col_counts[ent]), "pct": round(float(col_pct[ent]), 4)}
        for ent in birthright_entitlements
    }

    return {
        "birthright_entitlements": birthright_entitlements,
        "noise_entitlements": noise_entitlements,
        "filtered_matrix": filtered_matrix,
        "birthright_stats": birthright_stats,
    }