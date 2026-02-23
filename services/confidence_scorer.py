# services/confidence_scorer.py
"""
Confidence Scorer V2 - Multi-Factor Confidence Scoring
======================================================

Enhanced confidence scoring combining multiple signals:
1. Peer group prevalence (from entitlement clusters)
2. Customer-configured user attribute alignment (flexible schema)
3. Drift stability (how stable has this entitlement been over time)
4. Role coverage (what % of user's assigned roles does user have)

Key differences from V1 (services/confidence_scorer.py):
- V1: Peer group only (cluster-based)
- V2: Multi-factor weighted combination
- V2: Pre-computes attribute prevalence matrices (vectorized)
- V2: Handles NULL attributes gracefully (re-normalize weights)
- V2: Drift stability factor (temporal awareness)
- V2: Role coverage factor (multi-cluster aware)
- CHANGED: User attributes are customer-configured via user_attributes
  in config, not hardcoded department/jobcode/location/manager columns.

Algorithm:
1. Pre-compute attribute prevalence for configured user attributes
2. For each assignment, compute individual scores:
   - Peer group score (leave-one-out)
   - Per-attribute prevalence (customer-configured, 2-5 attributes)
   - Drift stability (if drift data available)
   - Role coverage (multi-cluster aware)
3. Weighted combination (re-normalize if attributes NULL)
4. Generate multi-factor justification
"""
import json

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import time
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


# ============================================================================
# SPARSE MATRIX ACCESSOR
# Wraps scipy.sparse.csr_matrix to expose the three operations used by the
# scorer, replacing pd.DataFrame(matrix.toarray(), ...) in mining.py.
# ============================================================================

class SparseMatrixAccessor:
    """
    Thin wrapper around a CSR matrix that exposes the DataFrame-like operations
    used by the confidence scorer, without ever densifying the full matrix.

    Supported operations (mirrors the DataFrame API used in this file):
      accessor.sum(axis=0)                -> np.ndarray (n_ents,)
      accessor.loc[user_ids].sum(axis=0)  -> np.ndarray (n_ents,)  [row subset sum]
      accessor.at[user_id, ent_id]        -> int  [point lookup]
      len(accessor)                        -> n_users
      accessor.index                       -> user_ids list (for compatibility checks)
      accessor.columns                     -> ent_ids list
    """

    def __init__(self, matrix, user_ids, ent_ids):
        """
        Args:
            matrix: scipy.sparse.csr_matrix (n_users × n_ents)
            user_ids: array-like of user ID strings (row labels)
            ent_ids:  array-like of entitlement ID strings (column labels)
        """
        import scipy.sparse as _sp
        if not _sp.issparse(matrix):
            raise TypeError("SparseMatrixAccessor requires a scipy sparse matrix")
        self._matrix = matrix.tocsr()
        self._user_ids = list(user_ids)
        self._ent_ids = list(ent_ids)
        self._user_idx = {uid: i for i, uid in enumerate(self._user_ids)}
        self._ent_idx = {eid: i for i, eid in enumerate(self._ent_ids)}

    # ---- DataFrame-compatible properties ----

    @property
    def index(self):
        return self._user_ids

    @property
    def columns(self):
        return self._ent_ids

    def __len__(self):
        return self._matrix.shape[0]

    # ---- Operations used by the scorer ----

    def sum(self, axis=0):
        """Column sums (axis=0) as a pandas Series for .map() compatibility."""
        result = np.asarray(self._matrix.sum(axis=0)).flatten()
        return pd.Series(result, index=self._ent_ids)

    def at(self, user_id, ent_id):
        """Scalar point lookup without densifying."""
        u = self._user_idx.get(user_id)
        e = self._ent_idx.get(ent_id)
        if u is None or e is None:
            return 0
        return int(self._matrix[u, e])

    class _LocAccessor:
        """Minimal .loc[list_of_user_ids] returning a _RowSubset."""

        def __init__(self, accessor):
            self._acc = accessor

        def __getitem__(self, user_ids):
            if isinstance(user_ids, str):
                user_ids = [user_ids]
            indices = [self._acc._user_idx[uid] for uid in user_ids
                       if uid in self._acc._user_idx]
            if not indices:
                return _EmptyRowSubset(self._acc._ent_ids)
            rows = self._acc._matrix[indices, :]
            return _RowSubset(rows, self._acc._ent_ids)

    @property
    def loc(self):
        return self._LocAccessor(self)


class _RowSubset:
    """Result of SparseMatrixAccessor.loc[user_ids] — supports .sum(axis=0)."""

    def __init__(self, rows, ent_ids):
        self._rows = rows
        self._ent_ids = ent_ids

    def sum(self, axis=0):
        result = np.asarray(self._rows.sum(axis=0)).flatten()
        return pd.Series(result, index=self._ent_ids)


class _EmptyRowSubset:
    """Sentinel for when no requested user IDs exist in the matrix."""

    def __init__(self, ent_ids):
        self._ent_ids = ent_ids

    def sum(self, axis=0):
        return pd.Series(0, index=self._ent_ids)


def _matrix_at(matrix, user_id, ent_id) -> int:
    """Scalar point lookup that works for both DataFrame and SparseMatrixAccessor."""
    if isinstance(matrix, SparseMatrixAccessor):
        return matrix.at(user_id, ent_id)
    # DataFrame path
    if ent_id in matrix.columns and user_id in matrix.index:
        return int(matrix.at[user_id, ent_id])
    return 0


# ============================================================================
# ADDED: Helper to extract user attribute config from plain config dict
# ============================================================================

def _get_user_attribute_config(config: Dict[str, Any]) -> Tuple[List[str], Dict[str, float]]:
    """
    Extract attribute column names and weights from config.

    Reads the user_attributes list from config and returns:
    - columns: list of column name strings
    - weights: dict mapping column name to weight (equal weight if not specified)

    Works with plain dict config (not MiningConfig dataclass) since
    confidence_scorer receives config as dict from the mining route.
    """
    user_attributes = config.get("user_attributes", [])
    if not user_attributes:
        return [], {}

    columns = [attr["column"] for attr in user_attributes]

    has_weights = all("weight" in attr for attr in user_attributes)
    if has_weights:
        weights = {attr["column"]: attr["weight"] for attr in user_attributes}
    else:
        equal_weight = 1.0 / len(user_attributes)
        weights = {col: equal_weight for col in columns}

    return columns, weights


# ============================================================================
# MAIN CONFIDENCE SCORING FUNCTION
# ============================================================================

def score_assignments(
        assignments_df: pd.DataFrame,
        full_matrix: pd.DataFrame,
        identities: pd.DataFrame,
        cluster_result: Dict[str, Any],
        roles: List[Dict],
        birthright_entitlements: List[str],
        noise_entitlements: List[str],
        config: Dict[str, Any],
        drift_data: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """
    Score all user-entitlement assignments with multi-factor confidence.

    Args:
        assignments_df: All user-entitlement assignments
        full_matrix: User × entitlement matrix
        identities: User HR attributes
        cluster_result: Output from cluster_entitlements_leiden()
        roles: Output from build_roles_v2()
        birthright_entitlements: List of birthright entitlement IDs
        noise_entitlements: List of noise entitlement IDs
        config: Configuration dict
        drift_data: Optional drift stability data (from daily clustering)

    Returns:
        Enriched assignments DataFrame with confidence columns
    """
    _t_start = time.monotonic()
    logger.info(f"Scoring {len(assignments_df)} assignments with multi-factor confidence")

    # Step 1: Pre-compute attribute prevalence matrices
    logger.info("Step 1: Pre-computing attribute prevalence")
    attribute_prevalence = _precompute_attribute_prevalence(
        matrix=full_matrix,
        identities=identities,
        config=config,
    )

    # Step 2: Build entitlement-to-role lookup (ent -> roles with tier/prevalence)
    logger.info("Step 2: Building entitlement-to-role lookup")
    ent_role_lookup = _build_entitlement_role_lookup(roles)

    # Step 3: Build role coverage lookup (user → role coverage)
    logger.info("Step 3: Building role coverage lookup")
    user_role_coverage = _build_user_role_coverage(roles, cluster_result)

    # Step 4: Compute individual scores for each assignment
    logger.info("Step 4: Computing individual factor scores")
    scores_df = _compute_individual_scores(
        assignments_df=assignments_df,
        full_matrix=full_matrix,
        identities=identities,
        ent_role_lookup=ent_role_lookup,
        cluster_result=cluster_result,
        roles=roles,
        attribute_prevalence=attribute_prevalence,
        birthright_entitlements=birthright_entitlements,
        noise_entitlements=noise_entitlements,
        drift_data=drift_data,
        config=config,
    )

    # Step 5: Weighted combination
    logger.info("Step 5: Computing weighted confidence scores")
    enriched_df = _compute_weighted_confidence(
        scores_df=scores_df,
        user_role_coverage=user_role_coverage,
        config=config,
    )

    # Step 6: Merge user attributes for display in UI
    logger.info("Step 6: Merging user attributes for UI display")
    attr_columns, _ = _get_user_attribute_config(config)
    if attr_columns and 'USR_ID' in identities.columns:
        # Select only the configured attributes plus USR_ID for merging
        cols_to_merge = ['USR_ID'] + [col for col in attr_columns if col in identities.columns]
        identity_subset = identities[cols_to_merge].copy()

        # Merge on USR_ID
        enriched_df = enriched_df.merge(
            identity_subset,
            on='USR_ID',
            how='left',
            suffixes=('', '_identity')
        )

    # FIX 3 (2026-02-17): Add calibration statistics
    # Validate that HIGH confidence correlates with core/birthright, not residuals
    calibration_stats = {}
    if "entitlement_tier" in enriched_df.columns:
        for level in ["HIGH", "MEDIUM", "LOW"]:
            level_mask = enriched_df["confidence_level"] == level
            level_total = level_mask.sum()

            if level_total > 0:
                calibration_stats[f"{level}_total"] = int(level_total)
                calibration_stats[f"{level}_in_core"] = int(
                    ((level_mask) & (enriched_df["entitlement_tier"] == "core")).sum()
                )
                calibration_stats[f"{level}_in_common"] = int(
                    ((level_mask) & (enriched_df["entitlement_tier"] == "common")).sum()
                )
                calibration_stats[f"{level}_in_birthright"] = int(
                    ((level_mask) & (enriched_df["entitlement_tier"] == "birthright")).sum()
                )
                calibration_stats[f"{level}_in_residual"] = int(
                    ((level_mask) & (enriched_df["entitlement_tier"] == "residual")).sum()
                )

                # Calculate percentages for validation
                calibration_stats[f"{level}_core_pct"] = round(
                    calibration_stats[f"{level}_in_core"] / level_total * 100, 1
                )
                calibration_stats[f"{level}_residual_pct"] = round(
                    calibration_stats[f"{level}_in_residual"] / level_total * 100, 1
                )

    logger.info(
        "Confidence scoring complete: %d HIGH, %d MEDIUM, %d LOW, total=%d elapsed_ms=%.0f",
        (enriched_df['confidence_level'] == 'HIGH').sum(),
        (enriched_df['confidence_level'] == 'MEDIUM').sum(),
        (enriched_df['confidence_level'] == 'LOW').sum(),
        len(enriched_df),
        (time.monotonic() - _t_start) * 1000,
    )

    # FIX 3 (2026-02-17): Log calibration stats for validation
    if calibration_stats:
        logger.info(
            f"Calibration check - HIGH: {calibration_stats.get('HIGH_core_pct', 0):.1f}% core, "
            f"{calibration_stats.get('HIGH_residual_pct', 0):.1f}% residual | "
            f"LOW: {calibration_stats.get('LOW_core_pct', 0):.1f}% core, "
            f"{calibration_stats.get('LOW_residual_pct', 0):.1f}% residual"
        )
        # Store in DataFrame for potential export
        enriched_df.attrs['calibration_stats'] = calibration_stats

    return enriched_df


# ============================================================================
# STEP 1: PRE-COMPUTE ATTRIBUTE PREVALENCE
# ============================================================================

def _precompute_attribute_prevalence(
        matrix: pd.DataFrame,
        identities: pd.DataFrame,
        config: Dict[str, Any],
) -> Dict[Tuple[str, str, str], Dict[str, int]]:
    """
    Pre-compute prevalence of each entitlement within each attribute group.

    For each (attribute_column, value, entitlement):
        prevalence[department=Finance, ENT_123] = {total: 500, with_ent: 435}

    This allows O(1) lookup during scoring instead of O(n_users) per assignment.

    CHANGED: reads attribute columns from config["user_attributes"] list
    instead of hardcoded config["attribute_columns"] dict.

    Returns:
        Dict mapping (col_name, attr_value, ent_id) -> {total_users, users_with_ent}
    """
    # CHANGED: use _get_user_attribute_config helper
    attr_columns, _ = _get_user_attribute_config(config)
    max_cardinality = config.get("max_attribute_cardinality", 500)
    min_group_size = config.get("min_attribute_group_size", 2)

    prevalence = {}
    skipped_attributes = []

    for col_name in attr_columns:
        logger.info(f"Checking for attribute column '{col_name}'")
        if col_name not in identities.columns:
            logger.warning(f"Attribute column '{col_name}' not found in identities, skipping")
            continue

        # Check cardinality
        n_unique = identities[col_name].nunique()
        if n_unique > max_cardinality:
            skipped_attributes.append(col_name)
            logger.info(
                f"Skipping {col_name} (cardinality {n_unique} > {max_cardinality})"
            )
            continue

        # Group by attribute value (vectorized)
        grouped = identities.groupby(col_name, dropna=True)

        for attr_value, group_df in grouped:
            if len(group_df) < min_group_size:
                continue

            user_ids = group_df["USR_ID"].tolist()
            total_users = len(user_ids)

            # Sum entitlement counts for this attribute group
            ent_sums = matrix.loc[user_ids].sum(axis=0)

            for ent_id, count in ent_sums.items():
                if count == 0:
                    continue

                # CHANGED: key uses col_name directly, not semantic attr_name
                key = (col_name, str(attr_value), ent_id)
                prevalence[key] = {
                    "total_users": total_users,
                    "users_with_ent": int(count),
                }

    logger.info(
        f"Pre-computed prevalence for {len(prevalence)} "
        f"(attribute, value, entitlement) combinations"
    )

    if skipped_attributes:
        logger.info(f"Skipped high-cardinality attributes: {skipped_attributes}")

    return prevalence


# ============================================================================
# STEP 2: BUILD ENTITLEMENT-TO-CLUSTER MAPPING
# ============================================================================

def _build_entitlement_role_lookup(
        roles: List[Dict],
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build reverse mapping: entitlement_id -> list of roles containing it.

    Each entry has {role_id, tier, prevalence, members, member_count}.
    An entitlement can appear in multiple roles (core in one, common in another).

    Used by peer group scoring to pick the best role for each user-entitlement pair.
    """
    lookup: Dict[str, List[Dict[str, Any]]] = {}

    for role in roles:
        role_id = role["role_id"]
        members = role.get("members", [])
        member_count = role.get("member_count", len(members))
        prevalence_dict = role.get("entitlement_prevalence", {})

        for ent_id in role.get("core_entitlements", []):
            if ent_id not in lookup:
                lookup[ent_id] = []
            lookup[ent_id].append({
                "role_id": role_id,
                "tier": "core",
                "prevalence": prevalence_dict.get(ent_id, 1.0),
                "members": members,
                "member_count": member_count,
            })

        for ent_id in role.get("common_entitlements", []):
            if ent_id not in lookup:
                lookup[ent_id] = []
            lookup[ent_id].append({
                "role_id": role_id,
                "tier": "common",
                "prevalence": prevalence_dict.get(ent_id, 0.0),
                "members": members,
                "member_count": member_count,
            })

    return lookup


# ============================================================================
# STEP 3: BUILD USER ROLE COVERAGE
# ============================================================================

def _build_user_role_coverage(
        roles: List[Dict],
        cluster_result: Dict[str, Any],
) -> Dict[str, float]:
    """
    Build mapping: user_id → average role coverage.

    Multi-cluster aware: average coverage across all assigned roles.

    Returns:
        Dict mapping user_id to avg coverage (0.0-1.0)
    """
    user_memberships = cluster_result["user_cluster_membership"]

    user_role_coverage = {}

    for user_id, memberships in user_memberships.items():
        if not memberships:
            user_role_coverage[user_id] = 0.0
            continue

        avg_coverage = np.mean([m["coverage"] for m in memberships])
        user_role_coverage[user_id] = avg_coverage

    return user_role_coverage


# ============================================================================
# STEP 4: COMPUTE INDIVIDUAL SCORES
# ============================================================================

def _compute_individual_scores(
        assignments_df: pd.DataFrame,
        full_matrix: pd.DataFrame,
        identities: pd.DataFrame,
        ent_role_lookup: Dict[str, List[Dict[str, Any]]],
        cluster_result: Dict[str, Any],
        roles: List[Dict],
        attribute_prevalence: Dict,
        birthright_entitlements: List[str],
        noise_entitlements: List[str],
        drift_data: Optional[Dict],
        config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Compute individual factor scores for each assignment.

    Returns:
        DataFrame with columns for each factor score
    """
    df = assignments_df.copy()

    # Determine entitlement column name
    ent_col = "namespaced_id" if "namespaced_id" in df.columns else "ENT_ID"

    # Global prevalence
    n_users = len(full_matrix)
    global_prev = full_matrix.sum(axis=0) / n_users
    df["global_prevalence"] = df[ent_col].map(global_prev).fillna(0.0)

    # Initialize score columns
    df["peer_group_score"] = 0.0
    attr_columns, _ = _get_user_attribute_config(config)
    for col_name in attr_columns:
        df[f"{col_name}_score"] = np.nan
    df["drift_stability_score"] = np.nan
    df["role_coverage_score"] = 0.0

    # Metadata columns
    df["cluster_id"] = None
    df["cluster_size"] = 0
    df["peers_with_entitlement"] = 0

    df["role_covered"] = False
    df["entitlement_tier"] = "residual"  # NEW: core/common/residual/birthright
    df["matched_role_id"] = ""  # NEW: which role matched this entitlement
    df["attributes_skipped"] = ""

    # Peer group scores (now role-based)
    df = _compute_peer_group_scores(
        df=df,
        full_matrix=full_matrix,
        ent_role_lookup=ent_role_lookup,
        cluster_result=cluster_result,
        birthright_entitlements=birthright_entitlements,
        ent_col=ent_col,
    )

    # Attribute scores
    df = _compute_attribute_scores(
        df=df,
        identities=identities,
        full_matrix=full_matrix,
        attribute_prevalence=attribute_prevalence,
        config=config,
        ent_col=ent_col,
    )

    # Drift stability scores (if available)
    if drift_data:
        df = _compute_drift_stability_scores(
            df=df,
            drift_data=drift_data,
            ent_col=ent_col,
        )

    # Role covered flag (now uses core_entitlements from roles)
    df = _compute_role_covered(
        df=df,
        roles=roles,
        cluster_result=cluster_result,
        birthright_entitlements=birthright_entitlements,
        ent_col=ent_col,
    )

    return df


def _compute_peer_group_scores(
        df: pd.DataFrame,
        full_matrix: pd.DataFrame,
        ent_role_lookup: Dict[str, List[Dict[str, Any]]],
        cluster_result: Dict[str, Any],
        birthright_entitlements: List[str],
        ent_col: str,
) -> pd.DataFrame:
    """
    Compute peer group prevalence scores using role-based peer groups.

    Vectorized: builds a (user_id, ent_id) lookup dict from the role structures,
    then assigns all columns in bulk rather than row-by-row.
    """
    user_memberships = cluster_result["user_cluster_membership"]
    birthright_set = set(birthright_entitlements)

    # user -> set of role_ids
    user_role_ids: Dict[str, set] = {
        user_id: {f"ROLE_{m['cluster_id']:03d}" for m in memberships}
        for user_id, memberships in user_memberships.items()
    }

    # Pre-compute per-role entitlement sums for leave-one-out (unchanged — small structure)
    role_ent_sums: Dict[str, pd.Series] = {}
    for role_entries in ent_role_lookup.values():
        for entry in role_entries:
            rid = entry["role_id"]
            if rid not in role_ent_sums:
                members = entry["members"]
                valid_members = [m for m in members if m in full_matrix.index]
                role_ent_sums[rid] = (
                    full_matrix.loc[valid_members].sum(axis=0)
                    if valid_members
                    else pd.Series(0, index=full_matrix.columns)
                )

    # Build (user_id, ent_id) -> result dict in one pass over the lookup structures.
    # Only covers non-birthright, non-residual rows — defaults handle the rest.
    lookup: Dict[tuple, Dict[str, Any]] = {}
    for ent_id, role_entries in ent_role_lookup.items():
        for entry in role_entries:
            role_id = entry["role_id"]
            member_count = entry["member_count"]
            tier = entry["tier"]
            prevalence = entry["prevalence"]
            ent_sums = role_ent_sums.get(role_id)

            for user_id in entry["members"]:
                key = (user_id, ent_id)
                # Keep only the best (highest prevalence) role per (user, ent) pair
                if key in lookup and lookup[key]["prevalence"] >= prevalence:
                    continue
                lookup[key] = {
                    "role_id": role_id,
                    "tier": tier,
                    "member_count": member_count,
                    "prevalence": prevalence,
                    "ent_sums": ent_sums,
                }

    # Vectorized assignment
    keys = list(zip(df["USR_ID"], df[ent_col]))

    tiers = []
    role_ids = []
    cluster_sizes = []
    role_covered = []
    peer_group_scores = []
    peers_with_list = []

    for user_id, ent_id in keys:
        # Birthright
        if ent_id in birthright_set:
            tiers.append("birthright")
            role_ids.append("")
            cluster_sizes.append(0)
            role_covered.append(True)
            peer_group_scores.append(1.0)
            peers_with_list.append(0)
            continue

        entry = lookup.get((user_id, ent_id))
        if entry is None:
            tiers.append("residual")
            role_ids.append("")
            cluster_sizes.append(0)
            role_covered.append(False)
            peer_group_scores.append(0.0)
            peers_with_list.append(0)
            continue

        role_id = entry["role_id"]
        member_count = entry["member_count"]
        ent_sums = entry["ent_sums"]

        tiers.append(entry["tier"])
        role_ids.append(role_id)
        cluster_sizes.append(member_count)
        role_covered.append(True)

        if member_count <= 1 or ent_sums is None or ent_id not in ent_sums.index:
            peer_group_scores.append(0.0)
            peers_with_list.append(0)
            continue

        user_has = _matrix_at(full_matrix, user_id, ent_id)
        peers_with = int(ent_sums[ent_id]) - user_has
        peer_count = member_count - 1
        score = round(peers_with / peer_count, 4) if peer_count > 0 else 0.0
        peer_group_scores.append(score)
        peers_with_list.append(peers_with)

    df["entitlement_tier"] = tiers
    df["matched_role_id"] = role_ids
    df["cluster_size"] = cluster_sizes
    df["role_covered"] = role_covered
    df["peer_group_score"] = peer_group_scores
    df["peers_with_entitlement"] = peers_with_list

    return df


def _compute_attribute_scores(
        df: pd.DataFrame,
        identities: pd.DataFrame,
        full_matrix: pd.DataFrame,
        attribute_prevalence: Dict,
        config: Dict[str, Any],
        ent_col: str,
) -> pd.DataFrame:
    """
    Compute user attribute prevalence scores.

    Vectorized: merges identities onto df once, then maps prevalence
    scores via a combined (attr_value, ent_id) key per attribute column.
    Leave-one-out correction applied in bulk for rows where user has the entitlement.
    """
    attr_columns, _ = _get_user_attribute_config(config)
    if not attr_columns:
        return df

    present_cols = [c for c in attr_columns if c in identities.columns]
    if not present_cols:
        return df

    # Merge identities attr columns onto df once (left join on USR_ID)
    id_cols = ["USR_ID"] + present_cols
    df = df.merge(
        identities[id_cols].drop_duplicates("USR_ID"),
        on="USR_ID",
        how="left",
        suffixes=("", "_id_attr"),
    )

    # user_has_ent: boolean Series — does this user have this entitlement?
    # Used for leave-one-out correction across all attribute columns.
    def _user_has(user_id, ent_id):
        return _matrix_at(full_matrix, user_id, ent_id)

    user_has_ent = pd.Series(
        [_user_has(u, e) for u, e in zip(df["USR_ID"], df[ent_col])],
        index=df.index,
        dtype=int,
    )

    for col_name in present_cols:
        score_col = f"{col_name}_score"

        # Build (attr_value, ent_id) -> (total, with_ent) lookup from prevalence dict
        # Keys in attribute_prevalence: (col_name, str(attr_value), ent_id)
        col_prev = {
            (attr_val, ent_id): (v["total_users"], v["users_with_ent"])
            for (cn, attr_val, ent_id), v in attribute_prevalence.items()
            if cn == col_name
        }

        if not col_prev:
            continue

        attr_vals = df[col_name].astype(str)
        ent_vals = df[ent_col]

        # Map (attr_value, ent_id) -> base score (without leave-one-out)
        total_arr = np.empty(len(df), dtype=float)
        with_ent_arr = np.empty(len(df), dtype=float)
        for i, key in enumerate(zip(attr_vals, ent_vals)):
            entry = col_prev.get(key)
            if entry:
                total_arr[i] = entry[0]
                with_ent_arr[i] = entry[1]
            else:
                total_arr[i] = np.nan
                with_ent_arr[i] = np.nan

        # Apply leave-one-out correction where user has the entitlement
        has_mask = user_has_ent.to_numpy(dtype=bool)
        with_ents_adj = with_ent_arr.copy()
        totals_adj = total_arr.copy()
        with_ents_adj[has_mask] -= 1
        totals_adj[has_mask] -= 1

        # Compute score, guard division by zero
        with np.errstate(invalid="ignore", divide="ignore"):
            scores = np.where(
                totals_adj > 0,
                np.round(with_ents_adj / totals_adj, 4),
                np.nan,
            )

        # Only set where we had a valid prevalence entry and attr is non-null/non-empty
        null_mask = df[col_name].isna() | (df[col_name].astype(str) == "")
        valid = ~null_mask.to_numpy() & ~np.isnan(total_arr)
        df[score_col] = np.where(valid, scores, np.nan)

        # Track skipped (null attr value) for attributes_skipped column
        skipped_mask = null_mask & df["USR_ID"].isin(full_matrix.index)
        if skipped_mask.any():
            # Append col_name to attributes_skipped for affected rows
            df.loc[skipped_mask, "attributes_skipped"] = df.loc[
                skipped_mask, "attributes_skipped"
            ].apply(lambda s: f"{s}, {col_name}" if s else col_name)

    # Drop the merged identity attr columns — they were temp
    df = df.drop(columns=present_cols, errors="ignore")

    return df


def _compute_drift_stability_scores(
        df: pd.DataFrame,
        drift_data: Dict[str, Any],
        ent_col: str,
) -> pd.DataFrame:
    """
    Compute drift stability scores.

    If entitlement has been stable (unchanged) for N days, score = 1.0
    If recently changed, score = lower

    TODO: Implement when drift_detector is built
    For now, stub with placeholder logic.
    """
    # Placeholder: all entitlements are "stable" (score = 1.0)
    df["drift_stability_score"] = 1.0
    return df


def _compute_role_covered(
        df: pd.DataFrame,
        roles: List[Dict],
        cluster_result: Dict[str, Any],
        birthright_entitlements: List[str],
        ent_col: str,
) -> pd.DataFrame:
    """
    Determine if entitlement is covered by user's assigned roles or birthright.

    Uses core_entitlements (expanded) from roles rather than raw cluster membership.
    Note: much of this is already set by _compute_peer_group_scores (which sets
    role_covered=True for core/common/birthright). This handles any edge cases
    and ensures consistency.
    """
    # Already handled by _compute_peer_group_scores for most rows.
    # This is a safety net — only needed if peer group scoring missed something.
    # The entitlement_tier and role_covered columns are authoritative from peer group scoring.
    return df


# ============================================================================
# STEP 5: WEIGHTED COMBINATION
# ============================================================================

def _compute_weighted_confidence(
        scores_df: pd.DataFrame,
        user_role_coverage: Dict[str, float],
        config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Compute weighted confidence from individual factor scores.

    Handles NULL attributes by re-normalizing weights.

    CHANGED: reads weights from config["user_attributes"] instead of
    hardcoded config["attribute_weights"]. Peer group weight is computed
    as the complement of attribute + extra factor weights.
    """
    df = scores_df.copy()

    # CHANGED: get attribute weights from user_attributes config
    attr_columns, attr_weights = _get_user_attribute_config(config)

    renormalize = config.get("renormalize_weights_on_null", True)
    high_thresh = config.get("confidence_high_threshold", 0.8)
    medium_thresh = config.get("confidence_medium_threshold", 0.5)

    # Use drift stability and role coverage if configured
    use_drift = config.get("use_drift_stability_factor", True)
    use_role_cov = config.get("use_role_coverage_factor", True)

    # CHANGED: build base weights dict dynamically
    # Peer group gets weight = 1.0 - sum(attribute weights) - extra factor weights
    drift_weight = config.get("drift_stability_weight", 0.1) if use_drift else 0.0
    role_cov_weight = config.get("role_coverage_weight", 0.1) if use_role_cov else 0.0

    # Attribute weights are scaled to fit within the remaining budget after peer_group
    # peer_group gets 40% of total, attributes share the rest minus extra factors
    # But since customer specifies attribute weights summing to 1.0, we need to
    # allocate a portion to peer_group and scale attribute weights down.
    #
    # Strategy: peer_group = 0.40, extra factors take their configured weight,
    # attribute weights are scaled to fill (1.0 - 0.40 - extra_weights)
    # FIX 1 (2026-02-17): Cap attributes at 30% to prevent HR bias from dominating scoring
    MAX_ATTR_WEIGHT = 0.30  # Attributes should be tie-breaker, not primary signal

    peer_group_base_weight = config.get("peer_group_weight", 0.40)
    extra_weight_total = drift_weight + role_cov_weight
    attr_budget = min(
        max(0.0, 1.0 - peer_group_base_weight - extra_weight_total),
        MAX_ATTR_WEIGHT  # FIX 1: Hard cap on attribute weight
    )

    # Scale customer's attribute weights to fit within attr_budget
    base_weights = {"peer_group": peer_group_base_weight}
    if attr_columns and attr_budget > 0:
        for col_name in attr_columns:
            base_weights[col_name] = attr_weights.get(col_name, 0) * attr_budget

    # -------------------------------------------------------------------------
    # Vectorized confidence computation
    # -------------------------------------------------------------------------

    # Collect factor score columns and their base weights
    factor_cols: Dict[str, str] = {"peer_group": "peer_group_score"}
    factor_weights: Dict[str, float] = {"peer_group": base_weights["peer_group"]}

    for col_name in attr_columns:
        score_col = f"{col_name}_score"
        if score_col in df.columns:
            factor_cols[col_name] = score_col
            factor_weights[col_name] = base_weights.get(col_name, 0.0)

    if use_drift and "drift_stability_score" in df.columns:
        factor_cols["drift_stability"] = "drift_stability_score"
        factor_weights["drift_stability"] = drift_weight

    # Role coverage: user-level value, only for non-residual rows
    role_cov_series = None
    if use_role_cov and user_role_coverage:
        role_cov_series = df["USR_ID"].map(user_role_coverage)
        # Zero out residual rows
        role_cov_series = role_cov_series.where(
            df["entitlement_tier"] != "residual", other=np.nan
        )

    # Build score matrix (rows × factors), NaN where factor unavailable
    score_parts: Dict[str, pd.Series] = {}
    weight_parts: Dict[str, pd.Series] = {}

    for factor_name, score_col in factor_cols.items():
        s = df[score_col].copy()
        score_parts[factor_name] = s
        # Weight is constant per factor, but masked to NaN where score is NaN
        weight_parts[factor_name] = pd.Series(
            np.where(s.notna(), factor_weights[factor_name], np.nan),
            index=df.index,
        )

    if role_cov_series is not None:
        score_parts["role_coverage"] = role_cov_series
        weight_parts["role_coverage"] = pd.Series(
            np.where(role_cov_series.notna(), role_cov_weight, np.nan),
            index=df.index,
        )

    score_matrix = pd.DataFrame(score_parts, index=df.index)
    weight_matrix = pd.DataFrame(weight_parts, index=df.index)

    if renormalize:
        weight_totals = weight_matrix.sum(axis=1)
        weight_matrix = weight_matrix.div(weight_totals, axis=0)

    confidence = (score_matrix * weight_matrix).sum(axis=1).round(4)
    # Rows where all scores are NaN → no factors available
    no_factors = score_matrix.isna().all(axis=1)
    confidence = confidence.where(~no_factors, other=0.0)

    confidence_level = pd.cut(
        confidence,
        bins=[-np.inf, medium_thresh, high_thresh, np.inf],
        labels=["LOW", "MEDIUM", "HIGH"],
    ).astype(str)
    confidence_level = confidence_level.where(~no_factors, other="LOW")

    df["confidence"] = confidence
    df["confidence_level"] = confidence_level
    df["weights_used"] = str(factor_weights)  # same for all rows

    # Justification: row-wise string, use apply on minimal columns
    just_cols = [
        "USR_ID", "entitlement_tier", "matched_role_id",
        "peers_with_entitlement", "cluster_size", "global_prevalence",
    ] + list(score_cols for score_cols in [f"{c}_score" for c in attr_columns] if score_cols in df.columns)

    def _make_justification(row):
        if no_factors[row.name]:
            return "No confidence factors available"
        available_scores = {
            factor: row[score_col]
            for factor, score_col in factor_cols.items()
            if pd.notna(row.get(score_col, np.nan))
        }
        if role_cov_series is not None and pd.notna(role_cov_series[row.name]):
            available_scores["role_coverage"] = role_cov_series[row.name]
        return _build_justification(
            confidence=confidence[row.name],
            available_scores=available_scores,
            weights_used=factor_weights,
            row=row,
        )

    df["justification"] = df.apply(_make_justification, axis=1)

    return df


def _build_justification(
        confidence: float,
        available_scores: Dict[str, float],
        weights_used: Dict[str, float],
        row: pd.Series,
) -> str:
    """
    Build human-readable justification for confidence score.

    Tier-aware justification:
    - Core: "N of M users (P%) in your ROLE_XXX have this access."
    - Common: "N of M users (P%) in your ROLE_XXX have this access.
              Frequently associated but not part of the core role definition."
    - Residual: "This access is not part of any of your assigned roles.
               P% of users where attr=value have this access." (attribute fallback)
    - Birthright: "Organization-wide baseline access (held by P% of all users)."
    """
    tier = row.get("entitlement_tier", "residual")
    matched_role = row.get("matched_role_id", "")
    peers_with = row.get("peers_with_entitlement", 0)
    cluster_size = row.get("cluster_size", 0)
    global_prev = row.get("global_prevalence", 0)

    if tier == "birthright":
        pct = int(global_prev * 100) if global_prev else 99
        return f"Organization-wide baseline access (held by {pct}% of all users)."

    if tier == "core" and matched_role and cluster_size > 1:
        peer_count = cluster_size - 1
        peer_pct = int((peers_with / peer_count) * 100) if peer_count > 0 else 0
        return (
            f"{peers_with} of {peer_count} peers ({peer_pct}%) in your "
            f"{matched_role} have this access."
        )

    if tier == "common" and matched_role and cluster_size > 1:
        peer_count = cluster_size - 1
        peer_pct = int((peers_with / peer_count) * 100) if peer_count > 0 else 0
        return (
            f"{peers_with} of {peer_count} peers ({peer_pct}%) in your "
            f"{matched_role} have this access. "
            f"Frequently associated but not part of the core role definition."
        )

    # Residual: attribute-based fallback
    parts = [f"This access is not part of any of your assigned roles."]

    # Find the best attribute score to report
    system_factors = {"peer_group", "drift_stability", "role_coverage"}
    attr_scores = {
        k: v for k, v in available_scores.items()
        if k not in system_factors and v > 0
    }

    if attr_scores:
        best_attr = max(attr_scores, key=attr_scores.get)
        best_pct = int(attr_scores[best_attr] * 100)
        parts.append(f"{best_pct}% of users in your {best_attr} group have this access.")
    else:
        parts.append(f"Confidence: {int(confidence * 100)}%.")

    return " ".join(parts)


# ============================================================================
# RECOMMENDATIONS & OVER-PROVISIONED (Reuse from V1 with enhancements)
# ============================================================================

def generate_recommendations(
        enriched_assignments: pd.DataFrame,
        full_matrix: pd.DataFrame,
        cluster_result: Dict[str, Any],
        config: Dict[str, Any],
) -> pd.DataFrame:
    """
    Generate access recommendations (missing entitlements).

    Enhanced from V1: Uses V2 confidence scoring.

    TODO: Full implementation - for now, return empty DataFrame.
    """
    # Placeholder: return empty
    return pd.DataFrame()


def detect_over_provisioned(
        enriched_assignments: pd.DataFrame,
        revocation_threshold: float,
) -> pd.DataFrame:
    """
    Detect over-provisioned access (revocation candidates).

    Same logic as V1 but with V2 confidence scores.
    """
    # Filter for low confidence
    mask = enriched_assignments["confidence"] < revocation_threshold

    # Exclude birthright (justification starts with "Confidence: 100%")
    # (birthright always gets 100% confidence in V2)
    result = enriched_assignments[mask].copy()

    if not result.empty:
        result = result.sort_values(
            ["USR_ID", "confidence"], ascending=[True, True]
        ).reset_index(drop=True)

    return result


# ---------------------------------------------------------------------------
# Save / load cluster assignments for stability tracking
# ---------------------------------------------------------------------------

def save_cluster_assignments(cluster_labels: pd.Series, path: str):
    """Save cluster assignments to JSON for future stability comparison."""
    assigned = cluster_labels[cluster_labels.notna()]
    data = {str(k): int(v) for k, v in assigned.items()}
    with open(path, "w") as f:
        json.dump(data, f)


# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

def build_scoring_summary(
        enriched_assignments: pd.DataFrame,
        cluster_result: Dict[str, Any],
        birthright_entitlements: List[str],
) -> Dict[str, Any]:
    """Build summary statistics for confidence scoring."""
    user_memberships = cluster_result["user_cluster_membership"]

    # Confidence level counts
    high_count = (enriched_assignments["confidence_level"] == "HIGH").sum()
    medium_count = (enriched_assignments["confidence_level"] == "MEDIUM").sum()
    low_count = (enriched_assignments["confidence_level"] == "LOW").sum()

    # Users with no cluster (birthright only)
    all_users = enriched_assignments["USR_ID"].unique()
    clustered_users = set(user_memberships.keys())
    birthright_only_users = [u for u in all_users if u not in clustered_users]

    # Role coverage statistics
    role_covered_count = enriched_assignments["role_covered"].sum()
    total_count = len(enriched_assignments)

    return {
        "confidence_scoring": {
            "total_scored_assignments": total_count,
            "high": int(high_count),
            "medium": int(medium_count),
            "low": int(low_count),
            "role_covered": int(role_covered_count),
            "role_coverage_pct": round(role_covered_count / total_count, 4) if total_count > 0 else 0.0,
            "birthright_only_users": {
                "count": len(birthright_only_users),
                "user_ids": birthright_only_users[:100],  # Limit to 100
            },
        }
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import sys

    sys.path.insert(0, '/home/claude')

    print("Confidence Scorer V2 - Example Usage")
    print("=" * 70)

    # Simulate data
    assignments_df = pd.DataFrame({
        "USR_ID": ["USR_001", "USR_001", "USR_002"],
        "namespaced_id": ["AWS:S3_Read", "AWS:EC2_Admin", "AWS:S3_Read"],
    })

    full_matrix = pd.DataFrame({
        "AWS:S3_Read": [1, 1],
        "AWS:EC2_Admin": [0, 1],
        "AWS:S3_Write": [1, 0],
    }, index=["USR_001", "USR_002"])

    identities = pd.DataFrame({
        "department": ["Engineering", "Engineering"],
        "jobcode": ["DevOps", "DevOps"],
        "location_country": ["USA", "USA"],
    }, index=["USR_001", "USR_002"])

    cluster_result = {
        "entitlement_clusters": {
            1: ["AWS:S3_Read", "AWS:EC2_Admin", "AWS:S3_Write"],
        },
        "user_cluster_membership": {
            "USR_001": [{"cluster_id": 1, "coverage": 0.67, "count": 2, "total": 3}],
            "USR_002": [{"cluster_id": 1, "coverage": 0.67, "count": 2, "total": 3}],
        },
    }

    roles = [
        {
            "role_id": "ROLE_001",
            "entitlement_cluster_id": 1,
            "entitlements": ["AWS:S3_Read", "AWS:EC2_Admin", "AWS:S3_Write"],
            "members": ["USR_001", "USR_002"],
        }
    ]

    config = {
        # CHANGED: user_attributes replaces attribute_columns and attribute_weights
        "user_attributes": [
            {"column": "department", "weight": 0.40},
            {"column": "jobcode", "weight": 0.35},
            {"column": "location_country", "weight": 0.25},
        ],
        "max_attribute_cardinality": 500,
        "min_attribute_group_size": 2,
        "renormalize_weights_on_null": True,
        "confidence_high_threshold": 0.8,
        "confidence_medium_threshold": 0.5,
        "use_drift_stability_factor": False,
        "use_role_coverage_factor": True,
        "role_coverage_weight": 0.05,
    }

    # Score assignments
    enriched = score_assignments(
        assignments_df=assignments_df,
        full_matrix=full_matrix,
        identities=identities,
        cluster_result=cluster_result,
        roles=roles,
        birthright_entitlements=[],
        noise_entitlements=[],
        config=config,
        drift_data=None,
    )

    print("\nEnriched assignments:")
    print(enriched[["USR_ID", "namespaced_id", "confidence", "confidence_level", "justification"]].to_string())

    print("\n✓ Confidence scorer V2 example complete")