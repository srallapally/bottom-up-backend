# services/confidence_scorer.py
"""
Confidence Scorer V2 - Multi-Factor Confidence Scoring
======================================================

Enhanced confidence scoring combining currently implemented signals:
1. Peer group prevalence (from entitlement clusters)
2. Customer-configured user attribute alignment (flexible schema)
3. Role coverage (what % of user's assigned roles does user have)

Planned but not implemented in the current release:
4. Drift stability (placeholder only; not currently used as a real signal)
5. Recommendation generation (placeholder only)

CHANGE 2026-04-02: Added tiered_birthright_roles support for sub-population
birthright scoring.

CHANGE 2026-04-02 (FIX-3): Tier-aware weight profiles. Core/common use
peer_group-dominant weights (attributes=0). Residual uses attribute-dominant
weights (peer_group=0). Fixes calibration bug where attribute dilution
caused 0% of core ents to score HIGH.

CHANGE 2026-04-02 (FIX-4): Assignment count dedup. Both
_compute_attribute_scores and score_assignments Step 6 merge on identities,
which can produce row duplication if identities has non-unique USR_ID.
Added explicit dedup guard after each merge.
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
# ============================================================================

class SparseMatrixAccessor:
    """
    Thin wrapper around a CSR matrix that exposes the DataFrame-like operations
    used by the confidence scorer, without ever densifying the full matrix.
    """

    def __init__(self, matrix, user_ids, ent_ids):
        import scipy.sparse as _sp
        if not _sp.issparse(matrix):
            raise TypeError("SparseMatrixAccessor requires a scipy sparse matrix")
        self._matrix = matrix.tocsr()
        self._user_ids = list(user_ids)
        self._ent_ids = list(ent_ids)
        self._user_idx = {uid: i for i, uid in enumerate(self._user_ids)}
        self._ent_idx = {eid: i for i, eid in enumerate(self._ent_ids)}

    @property
    def index(self):
        return self._user_ids

    @property
    def columns(self):
        return self._ent_ids

    def __len__(self):
        return self._matrix.shape[0]

    def sum(self, axis=0):
        result = np.asarray(self._matrix.sum(axis=0)).flatten()
        return pd.Series(result, index=self._ent_ids)

    def at(self, user_id, ent_id):
        u = self._user_idx.get(user_id)
        e = self._ent_idx.get(ent_id)
        if u is None or e is None:
            return 0
        return int(self._matrix[u, e])

    class _LocAccessor:
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
    def __init__(self, rows, ent_ids):
        self._rows = rows
        self._ent_ids = ent_ids

    def sum(self, axis=0):
        result = np.asarray(self._rows.sum(axis=0)).flatten()
        return pd.Series(result, index=self._ent_ids)


class _EmptyRowSubset:
    def __init__(self, ent_ids):
        self._ent_ids = ent_ids

    def sum(self, axis=0):
        return pd.Series(0, index=self._ent_ids)


def _matrix_at(matrix, user_id, ent_id) -> int:
    """Scalar point lookup that works for both DataFrame and SparseMatrixAccessor."""
    if isinstance(matrix, SparseMatrixAccessor):
        return matrix.at(user_id, ent_id)
    if ent_id in matrix.columns and user_id in matrix.index:
        return int(matrix.at[user_id, ent_id])
    return 0


def _get_user_attribute_config(config: Dict[str, Any]) -> Tuple[List[str], Dict[str, float]]:
    """Extract attribute column names and weights from config dict."""
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
        tiered_birthright_roles: Optional[List[Dict]] = None,
) -> pd.DataFrame:
    """Score all user-entitlement assignments with multi-factor confidence."""
    _t_start = time.monotonic()

    # FIX-4: Record input row count to detect duplication from merges
    _input_row_count = len(assignments_df)
    logger.info(f"Scoring {_input_row_count} assignments with multi-factor confidence")

    # Step 1: Pre-compute attribute prevalence matrices
    logger.info("Step 1: Pre-computing attribute prevalence")
    attribute_prevalence = _precompute_attribute_prevalence(
        matrix=full_matrix, identities=identities, config=config,
    )

    # Step 2: Build entitlement-to-role lookup
    logger.info("Step 2: Building entitlement-to-role lookup")
    ent_role_lookup = _build_entitlement_role_lookup(roles)

    # Step 3: Build role coverage lookup
    logger.info("Step 3: Building role coverage lookup")
    user_role_coverage = _build_user_role_coverage(roles, cluster_result)

    # Step 4: Compute individual factor scores
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
        tiered_birthright_roles=tiered_birthright_roles or [],
    )

    # FIX-4: Guard against row duplication from merges in Step 4
    if len(scores_df) != _input_row_count:
        logger.warning(
            "Row count changed after Step 4: %d -> %d. Deduplicating.",
            _input_row_count, len(scores_df),
        )
        ent_col = "namespaced_id" if "namespaced_id" in scores_df.columns else "ENT_ID"
        scores_df = scores_df.drop_duplicates(
            subset=["USR_ID", ent_col], keep="first",
        ).reset_index(drop=True)

    # Step 5: Weighted combination with tier-aware weight profiles
    logger.info("Step 5: Computing weighted confidence scores (tier-aware)")
    enriched_df = _compute_weighted_confidence(
        scores_df=scores_df,
        user_role_coverage=user_role_coverage,
        config=config,
    )

    # Step 6: Merge user attributes for display in UI
    logger.info("Step 6: Merging user attributes for UI display")
    attr_columns, _ = _get_user_attribute_config(config)
    if attr_columns and 'USR_ID' in identities.columns:
        cols_to_merge = ['USR_ID'] + [col for col in attr_columns if col in identities.columns]
        # FIX-4: Explicitly deduplicate identities before merge
        identity_subset = identities[cols_to_merge].drop_duplicates("USR_ID")

        _pre_merge_count = len(enriched_df)
        enriched_df = enriched_df.merge(
            identity_subset, on='USR_ID', how='left', suffixes=('', '_identity'),
        )
        # FIX-4: If merge produced duplicates, truncate back
        if len(enriched_df) != _pre_merge_count:
            logger.warning(
                "Step 6 merge produced row duplication: %d -> %d. Truncating.",
                _pre_merge_count, len(enriched_df),
            )
            enriched_df = enriched_df.head(_pre_merge_count).reset_index(drop=True)

    # Calibration statistics
    calibration_stats = {}
    if "entitlement_tier" in enriched_df.columns:
        for level in ["HIGH", "MEDIUM", "LOW"]:
            level_mask = enriched_df["confidence_level"] == level
            level_total = level_mask.sum()
            if level_total > 0:
                calibration_stats[f"{level}_total"] = int(level_total)
                for tier_name in ["core", "common", "birthright", "tiered_birthright", "residual"]:
                    calibration_stats[f"{level}_in_{tier_name}"] = int(
                        (level_mask & (enriched_df["entitlement_tier"] == tier_name)).sum()
                    )
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
    if calibration_stats:
        logger.info(
            f"Calibration check - HIGH: {calibration_stats.get('HIGH_core_pct', 0):.1f}% core, "
            f"{calibration_stats.get('HIGH_residual_pct', 0):.1f}% residual | "
            f"LOW: {calibration_stats.get('LOW_core_pct', 0):.1f}% core, "
            f"{calibration_stats.get('LOW_residual_pct', 0):.1f}% residual"
        )
        enriched_df.attrs['calibration_stats'] = calibration_stats

    return enriched_df


# ============================================================================
# STEP 1: PRE-COMPUTE ATTRIBUTE PREVALENCE
# ============================================================================

def _precompute_attribute_prevalence(
        matrix, identities: pd.DataFrame, config: Dict[str, Any],
) -> Dict[Tuple[str, str, str], Dict[str, int]]:
    """Pre-compute prevalence of each entitlement within each attribute group."""
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
        n_unique = identities[col_name].nunique()
        if n_unique > max_cardinality:
            skipped_attributes.append(col_name)
            logger.info(f"Skipping {col_name} (cardinality {n_unique} > {max_cardinality})")
            continue

        grouped = identities.groupby(col_name, dropna=True)
        for attr_value, group_df in grouped:
            if len(group_df) < min_group_size:
                continue
            user_ids = group_df["USR_ID"].tolist()
            total_users = len(user_ids)
            ent_sums = matrix.loc[user_ids].sum(axis=0)
            for ent_id, count in ent_sums.items():
                if count == 0:
                    continue
                prevalence[(col_name, str(attr_value), ent_id)] = {
                    "total_users": total_users, "users_with_ent": int(count),
                }

    logger.info(f"Pre-computed prevalence for {len(prevalence)} combinations")
    if skipped_attributes:
        logger.info(f"Skipped high-cardinality attributes: {skipped_attributes}")
    return prevalence


# ============================================================================
# STEP 2: BUILD ENTITLEMENT-TO-CLUSTER MAPPING
# ============================================================================

def _build_entitlement_role_lookup(roles: List[Dict]) -> Dict[str, List[Dict[str, Any]]]:
    """Build reverse mapping: entitlement_id -> list of roles containing it."""
    lookup: Dict[str, List[Dict[str, Any]]] = {}
    for role in roles:
        role_id = role["role_id"]
        members = role.get("members", [])
        member_count = role.get("member_count", len(members))
        prevalence_dict = role.get("entitlement_prevalence", {})

        for ent_id in role.get("core_entitlements", []):
            lookup.setdefault(ent_id, []).append({
                "role_id": role_id, "tier": "core",
                "prevalence": prevalence_dict.get(ent_id, 1.0),
                "members": members, "member_count": member_count,
            })
        for ent_id in role.get("common_entitlements", []):
            lookup.setdefault(ent_id, []).append({
                "role_id": role_id, "tier": "common",
                "prevalence": prevalence_dict.get(ent_id, 0.0),
                "members": members, "member_count": member_count,
            })
    return lookup


# ============================================================================
# STEP 3: BUILD USER ROLE COVERAGE
# ============================================================================

def _build_user_role_coverage(roles: List[Dict], cluster_result: Dict[str, Any]) -> Dict[str, float]:
    """Build mapping: user_id -> average role coverage."""
    user_memberships = cluster_result["user_cluster_membership"]
    user_role_coverage = {}
    for user_id, memberships in user_memberships.items():
        if not memberships:
            user_role_coverage[user_id] = 0.0
            continue
        user_role_coverage[user_id] = np.mean([m["coverage"] for m in memberships])
    return user_role_coverage


# ============================================================================
# STEP 4: COMPUTE INDIVIDUAL SCORES
# ============================================================================

def _compute_individual_scores(
        assignments_df, full_matrix, identities, ent_role_lookup, cluster_result,
        roles, attribute_prevalence, birthright_entitlements, noise_entitlements,
        drift_data, config, tiered_birthright_roles=None,
) -> pd.DataFrame:
    """Compute individual factor scores for each assignment."""
    df = assignments_df.copy()
    ent_col = "namespaced_id" if "namespaced_id" in df.columns else "ENT_ID"

    n_users = len(full_matrix)
    global_prev = full_matrix.sum(axis=0) / n_users
    df["global_prevalence"] = df[ent_col].map(global_prev).fillna(0.0)

    df["peer_group_score"] = 0.0
    attr_columns, _ = _get_user_attribute_config(config)
    for col_name in attr_columns:
        df[f"{col_name}_score"] = np.nan
    df["drift_stability_score"] = np.nan
    df["role_coverage_score"] = 0.0
    df["cluster_id"] = None
    df["cluster_size"] = 0
    df["peers_with_entitlement"] = 0
    df["role_covered"] = False
    df["entitlement_tier"] = "residual"
    df["matched_role_id"] = ""
    df["attributes_skipped"] = ""

    df = _compute_peer_group_scores(
        df=df, full_matrix=full_matrix, ent_role_lookup=ent_role_lookup,
        cluster_result=cluster_result, birthright_entitlements=birthright_entitlements,
        ent_col=ent_col, tiered_birthright_roles=tiered_birthright_roles or [],
        identities=identities,
    )
    df = _compute_attribute_scores(
        df=df, identities=identities, full_matrix=full_matrix,
        attribute_prevalence=attribute_prevalence, config=config, ent_col=ent_col,
    )
    if drift_data:
        df = _compute_drift_stability_scores(df=df, drift_data=drift_data, ent_col=ent_col)

    return df


def _compute_peer_group_scores(
        df, full_matrix, ent_role_lookup, cluster_result, birthright_entitlements,
        ent_col, tiered_birthright_roles=None, identities=None,
) -> pd.DataFrame:
    """Compute peer group prevalence scores using role-based peer groups."""
    user_memberships = cluster_result["user_cluster_membership"]
    birthright_set = set(birthright_entitlements)

    # Pre-compute per-role entitlement sums for leave-one-out
    role_ent_sums: Dict[str, pd.Series] = {}
    for role_entries in ent_role_lookup.values():
        for entry in role_entries:
            rid = entry["role_id"]
            if rid not in role_ent_sums:
                members = entry["members"]
                valid = [m for m in members if m in full_matrix.index]
                role_ent_sums[rid] = (
                    full_matrix.loc[valid].sum(axis=0)
                    if valid else pd.Series(0, index=full_matrix.columns)
                )

    # Build (user_id, ent_id) -> result dict
    lookup: Dict[tuple, Dict[str, Any]] = {}
    for ent_id, role_entries in ent_role_lookup.items():
        for entry in role_entries:
            for user_id in entry["members"]:
                key = (user_id, ent_id)
                if key in lookup and lookup[key]["prevalence"] >= entry["prevalence"]:
                    continue
                lookup[key] = {
                    "role_id": entry["role_id"], "tier": entry["tier"],
                    "member_count": entry["member_count"],
                    "prevalence": entry["prevalence"],
                    "ent_sums": role_ent_sums.get(entry["role_id"]),
                }

    # Tiered birthright lookup
    tiered_ent_to_roles: Dict[str, List[Dict]] = {}
    for tr in (tiered_birthright_roles or []):
        for eid in tr["entitlements"]:
            tiered_ent_to_roles.setdefault(eid, []).append(tr)

    user_attr_cache: Dict[str, Dict] = {}
    if tiered_ent_to_roles and identities is not None:
        _id_df = identities if "USR_ID" in identities.columns else identities.reset_index()
        user_attr_cache = _id_df.drop_duplicates("USR_ID").set_index("USR_ID").to_dict("index")

    keys = list(zip(df["USR_ID"], df[ent_col]))
    tiers, role_ids, cluster_sizes, role_covered_list = [], [], [], []
    peer_group_scores, peers_with_list, tiered_criteria_descs = [], [], []

    for user_id, ent_id in keys:
        if ent_id in birthright_set:
            tiers.append("birthright"); role_ids.append(""); cluster_sizes.append(0)
            role_covered_list.append(True); peer_group_scores.append(1.0)
            peers_with_list.append(0); tiered_criteria_descs.append("")
            continue

        # Tiered birthright check
        if ent_id in tiered_ent_to_roles and user_id in user_attr_cache:
            user_attrs = user_attr_cache[user_id]
            best_match = None
            for tr in tiered_ent_to_roles[ent_id]:
                c_col = list(tr["criteria"].keys())[0]
                c_val = list(tr["criteria"].values())[0]
                if str(user_attrs.get(c_col, "")) == c_val:
                    if best_match is None or tr["member_count"] < best_match["member_count"]:
                        best_match = tr
            if best_match is not None:
                prev = best_match["entitlement_prevalences"].get(ent_id, 0.0)
                tiers.append("tiered_birthright"); role_ids.append(best_match["role_id"])
                cluster_sizes.append(best_match["member_count"])
                role_covered_list.append(True); peer_group_scores.append(prev)
                peers_with_list.append(int(prev * best_match["member_count"]))
                tiered_criteria_descs.append(str(list(best_match["criteria"].values())[0]))
                continue

        entry = lookup.get((user_id, ent_id))
        if entry is None:
            tiers.append("residual"); role_ids.append(""); cluster_sizes.append(0)
            role_covered_list.append(False); peer_group_scores.append(0.0)
            peers_with_list.append(0); tiered_criteria_descs.append("")
            continue

        tiers.append(entry["tier"]); role_ids.append(entry["role_id"])
        cluster_sizes.append(entry["member_count"])
        role_covered_list.append(True); tiered_criteria_descs.append("")

        mc = entry["member_count"]; es = entry["ent_sums"]
        if mc <= 1 or es is None or ent_id not in es.index:
            peer_group_scores.append(0.0); peers_with_list.append(0); continue

        user_has = _matrix_at(full_matrix, user_id, ent_id)
        pw = int(es[ent_id]) - user_has
        pc = mc - 1
        peer_group_scores.append(round(pw / pc, 4) if pc > 0 else 0.0)
        peers_with_list.append(pw)

    df["entitlement_tier"] = tiers
    df["matched_role_id"] = role_ids
    df["cluster_size"] = cluster_sizes
    df["role_covered"] = role_covered_list
    df["peer_group_score"] = peer_group_scores
    df["peers_with_entitlement"] = peers_with_list
    df["tiered_criteria_desc"] = tiered_criteria_descs
    return df


def _compute_attribute_scores(df, identities, full_matrix, attribute_prevalence, config, ent_col):
    """Compute user attribute prevalence scores (vectorized with leave-one-out)."""
    attr_columns, _ = _get_user_attribute_config(config)
    if not attr_columns:
        return df
    present_cols = [c for c in attr_columns if c in identities.columns]
    if not present_cols:
        return df

    # FIX-4: Deduplicate identities before merge
    id_deduped = identities[["USR_ID"] + present_cols].drop_duplicates("USR_ID")
    _pre = len(df)
    df = df.merge(id_deduped, on="USR_ID", how="left", suffixes=("", "_id_attr"))
    if len(df) != _pre:
        logger.warning("_compute_attribute_scores merge: %d->%d rows. Truncating.", _pre, len(df))
        df = df.head(_pre).reset_index(drop=True)

    user_has_ent = pd.Series(
        [_matrix_at(full_matrix, u, e) for u, e in zip(df["USR_ID"], df[ent_col])],
        index=df.index, dtype=int,
    )

    for col_name in present_cols:
        score_col = f"{col_name}_score"
        col_prev = {
            (av, eid): (v["total_users"], v["users_with_ent"])
            for (cn, av, eid), v in attribute_prevalence.items() if cn == col_name
        }
        if not col_prev:
            continue

        attr_vals = df[col_name].astype(str)
        total_arr = np.empty(len(df), dtype=float)
        with_ent_arr = np.empty(len(df), dtype=float)
        for i, key in enumerate(zip(attr_vals, df[ent_col])):
            e = col_prev.get(key)
            if e:
                total_arr[i] = e[0]; with_ent_arr[i] = e[1]
            else:
                total_arr[i] = np.nan; with_ent_arr[i] = np.nan

        has_mask = user_has_ent.to_numpy(dtype=bool)
        wa = with_ent_arr.copy(); ta = total_arr.copy()
        wa[has_mask] -= 1; ta[has_mask] -= 1

        with np.errstate(invalid="ignore", divide="ignore"):
            scores = np.where(ta > 0, np.round(wa / ta, 4), np.nan)

        null_mask = df[col_name].isna() | (df[col_name].astype(str) == "")
        valid = ~null_mask.to_numpy() & ~np.isnan(total_arr)
        df[score_col] = np.where(valid, scores, np.nan)

        skipped_mask = null_mask & df["USR_ID"].isin(full_matrix.index)
        if skipped_mask.any():
            df.loc[skipped_mask, "attributes_skipped"] = df.loc[
                skipped_mask, "attributes_skipped"
            ].apply(lambda s: f"{s}, {col_name}" if s else col_name)

    df = df.drop(columns=present_cols, errors="ignore")
    return df


def _compute_drift_stability_scores(df, drift_data, ent_col):
    # NOT IMPLEMENTED — returns 1.0 (maximum stability) for every entitlement.
    # This value is currently never used in practice because the call site at line 409
    # gates on `drift_data` being non-None, and the renormalize path drops the column
    # before it influences the final score.  Replace this stub before enabling drift
    # stability weighting in production.
    logger.warning(
        "_compute_drift_stability_scores called with real drift_data but is not "
        "implemented; all entitlements will receive drift_stability_score=1.0"
    )
    df["drift_stability_score"] = 1.0
    return df


# ============================================================================
# STEP 5: WEIGHTED COMBINATION — TIER-AWARE WEIGHT PROFILES
# ============================================================================

def _compute_weighted_confidence(scores_df, user_role_coverage, config):
    """
    FIX-3: Tier-aware weight profiles.

    - core/common: peer_group=0.80, role_cov=0.10, drift=0.10, attributes=0.0
    - residual: attributes=0.80, role_cov=0.10, drift=0.10, peer_group=0.0
    - birthright/tiered_birthright: direct override (1.0 or sub-pop prevalence)

    Note: drift weight is defined but drift_data is always None in the current
    pipeline. The renormalize path excludes it from the weighted sum. Effective
    weights are 0.80 peer_group + 0.10 role_cov (renormalized to ~0.89/0.11)
    for core/common.
    """
    df = scores_df.copy()
    attr_columns, attr_weights = _get_user_attribute_config(config)
    renormalize = config.get("renormalize_weights_on_null", True)
    high_thresh = config.get("confidence_high_threshold", 0.8)
    medium_thresh = config.get("confidence_medium_threshold", 0.5)
    use_drift = config.get("use_drift_stability_factor", True)
    use_role_cov = config.get("use_role_coverage_factor", True)

    # Tier-aware weight profiles
    core_w = {"peer_group": 0.80}
    if use_role_cov: core_w["role_coverage"] = 0.10
    if use_drift: core_w["drift_stability"] = 0.10

    res_w = {"peer_group": 0.0}
    if attr_columns:
        for c in attr_columns:
            res_w[c] = attr_weights.get(c, 0) * 0.80
    if use_role_cov: res_w["role_coverage"] = 0.10
    if use_drift: res_w["drift_stability"] = 0.10

    tier_s = df["entitlement_tier"]
    is_cc = tier_s.isin(["core", "common"])
    is_res = tier_s == "residual"
    is_br = tier_s == "birthright"
    is_tbr = tier_s == "tiered_birthright"

    factor_cols: Dict[str, str] = {"peer_group": "peer_group_score"}
    for c in attr_columns:
        sc = f"{c}_score"
        if sc in df.columns:
            factor_cols[c] = sc
    if use_drift and "drift_stability_score" in df.columns:
        factor_cols["drift_stability"] = "drift_stability_score"

    role_cov_series = None
    if use_role_cov and user_role_coverage:
        role_cov_series = df["USR_ID"].map(user_role_coverage)
        role_cov_series = role_cov_series.where(~is_res, other=np.nan)

    score_parts, weight_parts = {}, {}
    for fn, sc in factor_cols.items():
        s = df[sc].copy()
        score_parts[fn] = s
        w = pd.Series(np.nan, index=df.index)
        w[is_cc] = core_w.get(fn, 0.0)
        w[is_res] = res_w.get(fn, 0.0)
        w[is_br | is_tbr] = 0.0
        w = w.where(s.notna(), other=np.nan)
        w = w.where(w > 0, other=np.nan)
        weight_parts[fn] = w

    if role_cov_series is not None:
        score_parts["role_coverage"] = role_cov_series
        w = pd.Series(np.nan, index=df.index)
        w[is_cc] = core_w.get("role_coverage", 0.0)
        w[is_res] = res_w.get("role_coverage", 0.0)
        w[is_br | is_tbr] = 0.0
        w = w.where(role_cov_series.notna(), other=np.nan)
        w = w.where(w > 0, other=np.nan)
        weight_parts["role_coverage"] = w

    sm = pd.DataFrame(score_parts, index=df.index)
    wm = pd.DataFrame(weight_parts, index=df.index)

    if renormalize:
        wt = wm.sum(axis=1).where(lambda x: x > 0, other=1.0)
        wm = wm.div(wt, axis=0)

    confidence = (sm * wm).sum(axis=1).round(4)
    no_factors = sm.isna().all(axis=1)
    confidence = confidence.where(~no_factors, other=0.0)

    # Override: birthright=1.0, tiered_birthright=sub-pop prevalence
    confidence[is_br] = 1.0
    confidence[is_tbr] = df.loc[is_tbr, "peer_group_score"]

    confidence_level = pd.cut(
        confidence, bins=[-np.inf, medium_thresh, high_thresh, np.inf],
        labels=["LOW", "MEDIUM", "HIGH"],
    ).astype(str)
    confidence_level = confidence_level.where(~no_factors, other="LOW")

    df["confidence"] = confidence
    df["confidence_level"] = confidence_level
    df["weights_used"] = "residual"
    df.loc[is_cc, "weights_used"] = "core_common"
    df.loc[is_br, "weights_used"] = "birthright_override"
    df.loc[is_tbr, "weights_used"] = "tiered_br_override"

    # Justification
    def _just(row):
        if no_factors[row.name]:
            return "No confidence factors available"
        avail = {f: row[sc] for f, sc in factor_cols.items() if pd.notna(row.get(sc, np.nan))}
        if role_cov_series is not None and pd.notna(role_cov_series[row.name]):
            avail["role_coverage"] = role_cov_series[row.name]
        return _build_justification(confidence[row.name], avail, row)

    df["justification"] = df.apply(_just, axis=1)
    return df


def _build_justification(confidence, available_scores, row):
    """Build human-readable justification for confidence score."""
    tier = row.get("entitlement_tier", "residual")
    matched_role = row.get("matched_role_id", "")
    peers_with = row.get("peers_with_entitlement", 0)
    cluster_size = row.get("cluster_size", 0)
    global_prev = row.get("global_prevalence", 0)

    if tier == "birthright":
        pct = int(global_prev * 100) if global_prev else 99
        return f"Organization-wide baseline access (held by {pct}% of all users)."

    if tier == "tiered_birthright":
        subpop_desc = row.get("tiered_criteria_desc", "") or "sub-population"
        if cluster_size > 0 and peers_with:
            pct = int((peers_with / cluster_size) * 100)
            return f"Sub-population baseline: {pct}% of {subpop_desc} staff have this entitlement."
        return f"Sub-population baseline access for {subpop_desc}."

    if tier == "core" and matched_role and cluster_size > 1:
        pc = cluster_size - 1
        pp = int((peers_with / pc) * 100) if pc > 0 else 0
        return f"{peers_with} of {pc} peers ({pp}%) in your {matched_role} have this access."

    if tier == "common" and matched_role and cluster_size > 1:
        pc = cluster_size - 1
        pp = int((peers_with / pc) * 100) if pc > 0 else 0
        return (f"{peers_with} of {pc} peers ({pp}%) in your {matched_role} have this access. "
                f"Frequently associated but not part of the core role definition.")

    parts = ["This access is not part of any of your assigned roles."]
    sys_factors = {"peer_group", "drift_stability", "role_coverage"}
    attr_scores = {k: v for k, v in available_scores.items() if k not in sys_factors and v > 0}
    if attr_scores:
        best = max(attr_scores, key=attr_scores.get)
        parts.append(f"{int(attr_scores[best] * 100)}% of users in your {best} group have this access.")
    else:
        parts.append(f"Confidence: {int(confidence * 100)}%.")
    return " ".join(parts)


# ============================================================================
# RECOMMENDATIONS & OVER-PROVISIONED
# ============================================================================

def generate_recommendations(enriched_assignments, full_matrix, cluster_result, config):
    r"""
    Placeholder for future access recommendation generation.

    Current behavior:
        returns an empty DataFrame with a stable schema marker so callers can
        distinguish 'not implemented' from 'implemented but no recommendations'.
    """
    return pd.DataFrame(columns=["status", "message"]).assign(
        status=["not_implemented"],
        message=["Recommendation generation is not implemented in the current release"],
    )

def detect_over_provisioned(enriched_assignments, revocation_threshold):
    mask = enriched_assignments["confidence"] < revocation_threshold
    result = enriched_assignments[mask].copy()
    if not result.empty:
        result = result.sort_values(["USR_ID", "confidence"], ascending=[True, True]).reset_index(drop=True)
    return result

def save_cluster_assignments(cluster_labels, path):
    assigned = cluster_labels[cluster_labels.notna()]
    with open(path, "w") as f:
        json.dump({str(k): int(v) for k, v in assigned.items()}, f)

def build_scoring_summary(enriched_assignments, cluster_result, birthright_entitlements):
    um = cluster_result["user_cluster_membership"]
    h = (enriched_assignments["confidence_level"] == "HIGH").sum()
    m = (enriched_assignments["confidence_level"] == "MEDIUM").sum()
    lo = (enriched_assignments["confidence_level"] == "LOW").sum()
    all_u = enriched_assignments["USR_ID"].unique()
    br_only = [u for u in all_u if u not in um]
    rc = enriched_assignments["role_covered"].sum()
    t = len(enriched_assignments)
    return {"confidence_scoring": {
        "total_scored_assignments": t, "high": int(h), "medium": int(m), "low": int(lo),
        "role_covered": int(rc),
        "role_coverage_pct": round(rc / t, 4) if t > 0 else 0.0,
        "birthright_only_users": {"count": len(br_only), "user_ids": br_only[:100]},
    }}