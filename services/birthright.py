# services/birthright.py
# CHANGE 2026-02-17: Added scipy.sparse import for sparse matrix support
# CHANGE 2026-04-02: Added detect_tiered_birthrights() for sub-population birthrights
import pandas as pd
import numpy as np
import time
import logging
from scipy.sparse import csr_matrix


logger = logging.getLogger(__name__)


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

    logger.info(
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


# ============================================================================
# TIERED BIRTHRIGHT DETECTION (Step 4.5)
# Added 2026-04-02 per tiered_birthright_design_doc.md
# ============================================================================

def detect_tiered_birthrights(
        matrix,              # csr_matrix (users × ents), universal birthrights already removed
        user_ids,            # array of user ID strings
        ent_ids,             # array of entitlement ID strings
        identities,          # DataFrame with USR_ID column + HR attribute columns
        user_attributes,     # list of {"column": str, "weight": float} from config
        threshold=0.80,      # minimum prevalence within sub-population
        min_subpop_size=50,  # minimum sub-population size
):
    """
    Detect entitlements that are baseline access within HR-defined sub-populations.

    Inserted as Step 4.5 between universal birthright detection and Leiden clustering.
    Zeros out matrix cells for (sub-population users × tiered birthright entitlements)
    so Leiden clustering operates on a cleaner co-occurrence graph.

    Args:
        matrix: csr_matrix (users × ents), universal birthrights/noise already removed
        user_ids: array-like of user IDs (row labels)
        ent_ids: array-like of entitlement IDs (column labels)
        identities: DataFrame with USR_ID and HR attribute columns
        user_attributes: list of {"column": str, ...} from config
        threshold: minimum prevalence within sub-population (default 0.80)
        min_subpop_size: minimum sub-population size (default 50)

    Returns:
        dict with:
            - tiered_birthright_roles: list of role dicts
            - filtered_matrix: csr_matrix with sub-population cells zeroed
            - filtered_ent_ids: same as input ent_ids (columns unchanged)
            - tiered_stats: summary statistics
    """
    _t_start = time.monotonic()

    user_ids_array = np.array(user_ids)
    ent_ids_array = np.array(ent_ids)

    # Build user_id -> row index mapping
    user_idx_map = {uid: i for i, uid in enumerate(user_ids_array)}

    # Build ent_id -> col index mapping
    ent_idx_map = {eid: i for i, eid in enumerate(ent_ids_array)}

    # Index identities by USR_ID for fast lookup
    if "USR_ID" not in identities.columns:
        identities = identities.reset_index()
    id_indexed = identities.set_index("USR_ID")

    tiered_roles = []
    sub_pops_scanned = 0
    total_pairs_absorbed = 0

    for attr_config in user_attributes:
        column = attr_config["column"]

        if column not in id_indexed.columns:
            logger.warning(
                "tiered_birthright: attribute column '%s' not in identities, skipping",
                column,
            )
            continue

        # Track entitlements already assigned within THIS attribute
        # (within-attribute de-duplication per design doc §4.4)
        assigned_within_attribute = set()

        # Get distinct values sorted by sub-population size descending
        # (larger sub-populations take priority within the same attribute)
        value_counts = id_indexed[column].value_counts()

        for value in value_counts.index:
            # Get user IDs in this sub-population that exist in the matrix
            sub_pop_user_ids = id_indexed.index[id_indexed[column] == value]
            sub_pop_indices = np.array([
                user_idx_map[uid] for uid in sub_pop_user_ids
                if uid in user_idx_map
            ], dtype=np.int32)

            sub_pop_size = len(sub_pop_indices)
            sub_pops_scanned += 1

            if sub_pop_size < min_subpop_size:
                continue

            # Compute entitlement prevalence within sub-population (sparse)
            sub_matrix = matrix[sub_pop_indices]
            ent_counts = np.asarray(sub_matrix.sum(axis=0)).flatten()
            prevalence = ent_counts / sub_pop_size

            # Find qualifying entitlements (within-attribute dedup)
            qualifying_ents = []
            qualifying_prevalences = {}

            for i in range(len(ent_ids_array)):
                if prevalence[i] >= threshold:
                    eid = ent_ids_array[i]
                    if eid not in assigned_within_attribute:
                        qualifying_ents.append(eid)
                        qualifying_prevalences[eid] = round(float(prevalence[i]), 4)

            if not qualifying_ents:
                continue

            tiered_roles.append({
                "role_id": f"BIRTHRIGHT_{column}_{value}",
                "type": "tiered_birthright",
                "criteria": {column: str(value)},
                "entitlements": qualifying_ents,
                "member_count": sub_pop_size,
                "entitlement_prevalences": qualifying_prevalences,
            })

            # Mark these entitlements as assigned within this attribute
            assigned_within_attribute.update(qualifying_ents)

    # Build filtered matrix: zero out cells for (sub-pop user × tiered ent)
    if tiered_roles:
        from scipy.sparse import lil_matrix as _lil_matrix

        filtered = _lil_matrix(matrix.shape, dtype=matrix.dtype)
        filtered[:] = matrix

        for role in tiered_roles:
            col_key = list(role["criteria"].keys())[0]
            col_val = list(role["criteria"].values())[0]

            # Get sub-population user indices
            sub_pop_user_ids = id_indexed.index[id_indexed[col_key] == col_val]
            sub_pop_row_indices = [
                user_idx_map[uid] for uid in sub_pop_user_ids
                if uid in user_idx_map
            ]

            # Get entitlement column indices
            ent_col_indices = [
                ent_idx_map[eid] for eid in role["entitlements"]
                if eid in ent_idx_map
            ]

            pairs_zeroed = 0
            for u_idx in sub_pop_row_indices:
                for e_idx in ent_col_indices:
                    if filtered[u_idx, e_idx] != 0:
                        filtered[u_idx, e_idx] = 0
                        pairs_zeroed += 1

            total_pairs_absorbed += pairs_zeroed

        filtered_matrix = filtered.tocsr()
        filtered_matrix.eliminate_zeros()
    else:
        filtered_matrix = matrix.copy()

    unique_ents = set()
    for role in tiered_roles:
        unique_ents.update(role["entitlements"])

    elapsed_ms = (time.monotonic() - _t_start) * 1000
    logger.info(
        "tiered birthright detection complete: sub_pops_scanned=%d roles=%d "
        "unique_ents=%d pairs_absorbed=%d elapsed_ms=%.0f",
        sub_pops_scanned,
        len(tiered_roles),
        len(unique_ents),
        total_pairs_absorbed,
        elapsed_ms,
    )

    return {
        "tiered_birthright_roles": tiered_roles,
        "filtered_matrix": filtered_matrix,
        "filtered_ent_ids": list(ent_ids_array),
        "tiered_stats": {
            "sub_populations_scanned": sub_pops_scanned,
            "sub_populations_with_birthrights": len(tiered_roles),
            "total_entitlements_extracted": len(unique_ents),
            "total_assignment_pairs_absorbed": total_pairs_absorbed,
        },
    }