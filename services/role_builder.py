# services/role_builder.py
"""
Role Builder V2 - Two-Tier Prevalence Role Construction
========================================================

Builds role definitions from Leiden entitlement clusters with:
- Two-tier entitlement classification (core / common)
- Sparse matrix multiply for prevalence computation (Option 3)
- Birthright promotion detection
- Pairwise role overlap / merge candidate detection
- Multi-membership support (user can belong to 1-5 roles)

Algorithm:
1. Seed from Leiden clusters (seed entitlements + members)
2. Build sparse membership matrix R (users x roles)
3. Sparse matrix multiply: prevalence = (R.T @ M) / role_sizes
4. Classify entitlements into core/common tiers per role
5. Birthright promotion scan (core in >N% of roles)
6. Pairwise role overlap (Jaccard on core sets, flag merge candidates)
7. Enrich roles (HR summary, catalog, coverage distribution)
8. Residuals against expanded core entitlement sets
9. Build birthright role, summary, multi-cluster stats
"""

import numpy as np
import pandas as pd
import scipy.sparse as sp
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# MAIN ROLE BUILDING FUNCTION
# ============================================================================

def build_roles(
        matrix,  # CHANGE 2026-02-17: Now csr_matrix
        user_ids,  # CHANGE 2026-02-17: User IDs (row labels)
        ent_ids,  # CHANGE 2026-02-17: Entitlement IDs (column labels)
        cluster_result: Dict[str, Any],
        birthright_result: Dict[str, Any],
        identities: pd.DataFrame,
        catalog: Optional[pd.DataFrame] = None,
        config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build role definitions from entitlement clusters with two-tier prevalence.

    CHANGE 2026-02-17: Updated to work with sparse matrix + indices.

    Args:
        matrix: scipy.sparse.csr_matrix (users × entitlements)
        user_ids: array-like of user IDs (row labels)
        ent_ids: array-like of entitlement IDs (column labels)
        cluster_result: Output from cluster_entitlements_leiden()
        birthright_result: Output from detect_birthright()
        identities: User HR attributes DataFrame
        catalog: Optional entitlement catalog (for enrichment)
        config: Config dict with user_attributes and prevalence thresholds

    Returns:
        dict with:
            - roles: List of role objects with core/common tiers
            - birthright_role: Baseline access role
            - birthright_promotions: Entitlements flagged for escalation
            - merge_candidates: Role pairs with high core overlap
            - residuals: Per-user entitlements not covered by core sets
            - summary: Overall statistics
            - multi_cluster_info: Statistics about multi-membership
    """
    config = config or {}

    logger.info("Building V2 roles with two-tier prevalence")

    entitlement_clusters = cluster_result["entitlement_clusters"]
    user_memberships = cluster_result["user_cluster_membership"]
    cluster_metadata = cluster_result["cluster_metadata"]

    birthright_ents = set(birthright_result["birthright_entitlements"])
    noise_ents = set(birthright_result.get("noise_entitlements", []))
    excluded_ents = birthright_ents | noise_ents

    # ---- Step 1: Build seed roles from clusters ----
    min_entitlements = config.get("min_entitlements_per_role", 2)
    logger.info(f"Step 1: Building seed roles from {len(entitlement_clusters)} clusters "
                f"(min_entitlements={min_entitlements})")

    roles = []
    filtered_clusters = {"low_entitlements": 0, "zero_members": 0, "details": []}

    for cluster_id in sorted(entitlement_clusters.keys()):
        cluster_ents = entitlement_clusters[cluster_id]

        if len(cluster_ents) < min_entitlements:
            filtered_clusters["low_entitlements"] += 1
            filtered_clusters["details"].append({
                "cluster_id": cluster_id,
                "reason": "low_entitlements",
                "entitlement_count": len(cluster_ents),
                "entitlements": cluster_ents,
            })
            continue

        # Collect members for this cluster
        members = []
        member_coverage_data = {}
        for user_id, memberships in user_memberships.items():
            for membership in memberships:
                if membership["cluster_id"] == cluster_id:
                    members.append(user_id)
                    member_coverage_data[user_id] = {
                        "coverage": membership["coverage"],
                        "has_count": membership["count"],
                        "total_count": membership["total"],
                    }
                    break

        if not members:
            filtered_clusters["zero_members"] += 1
            filtered_clusters["details"].append({
                "cluster_id": cluster_id,
                "reason": "zero_members",
                "entitlement_count": len(cluster_ents),
            })
            continue

        meta = cluster_metadata.get(cluster_id, {})

        roles.append({
            "role_id": f"ROLE_{cluster_id:03d}",
            "role_name": f"ROLE_{cluster_id:03d}",
            "seed_cluster_id": cluster_id,
            "seed_entitlements": list(cluster_ents),
            "members": members,
            "member_count": len(members),
            "member_coverage": member_coverage_data,
            "avg_coverage": meta.get("avg_coverage", 0.0),
            "min_coverage": meta.get("min_coverage", 0.0),
            "max_coverage": meta.get("max_coverage", 0.0),
        })

    logger.info(f"Step 1 complete: {len(roles)} seed roles")

    # ---- Step 2-3: Sparse prevalence computation ----
    logger.info("Step 2-3: Computing prevalence via sparse matrix multiply")
    # CHANGE 2026-02-17: Pass user_ids and ent_ids
    prevalence_df = _compute_prevalence_matrix(roles, matrix, user_ids, ent_ids, excluded_ents)

    # ---- Step 4: Classify entitlements into tiers ----
    logger.info("Step 4: Classifying entitlements into core/common tiers")
    _classify_entitlement_tiers(roles, prevalence_df, config)

    # ---- Step 5: Birthright promotion scan ----
    logger.info("Step 5: Scanning for birthright promotion candidates")
    birthright_promotions = _detect_birthright_promotions(roles, config)

    if birthright_promotions:
        logger.info(f"Found {len(birthright_promotions)} birthright promotion candidates")

    # ---- Step 6: Pairwise role overlap ----
    logger.info("Step 6: Computing pairwise role overlap")
    merge_candidates = _compute_role_overlap(roles, config)

    if merge_candidates:
        logger.info(f"Found {len(merge_candidates)} merge candidates")

    # ---- Step 7: Enrich roles ----
    logger.info("Step 7: Enriching roles with HR summary and catalog")
    for role in roles:
        role["hr_summary"] = _summarize_hr_attributes(
            role["members"], identities, config
        )
        # Enrich all entitlements that appear in core + common
        all_role_ents = role["core_entitlements"] + role["common_entitlements"]
        role["entitlements_detail"] = _enrich_entitlements(all_role_ents, catalog)
        role["coverage_distribution"] = _compute_coverage_distribution(
            role["member_coverage"]
        )
        # Backward compat: "entitlements" field = core entitlements
        role["entitlements"] = role["core_entitlements"]
        role["entitlement_count"] = len(role["core_entitlements"])

    # ---- Step 8: Residuals against expanded core sets ----
    logger.info("Step 8: Computing residual access against core entitlement sets")
    # CHANGE 2026-02-17: Pass user_ids and ent_ids
    residuals = _compute_residuals_v2(
        matrix=matrix,
        user_ids=user_ids,
        ent_ids=ent_ids,
        roles=roles,
        user_memberships=user_memberships,
        birthright_entitlements=list(birthright_ents),
    )

    # ---- Step 9: Birthright role, summary, stats ----
    logger.info("Step 9: Building birthright role and summary statistics")
    birthright_role = _build_birthright_role(
        birthright_result=birthright_result,
        catalog=catalog,
    )

    multi_cluster_info = _compute_multi_cluster_stats(
        user_memberships=user_memberships,
        entitlement_clusters=entitlement_clusters,
    )

    naming_summary = _compute_naming_summary(roles)

    summary = _build_summary(
        roles=roles,
        user_memberships=user_memberships,
        birthright_result=birthright_result,
        residuals=residuals,
        matrix=matrix,
        user_ids=user_ids,  # CHANGE 2026-02-17
    )

    total_filtered = filtered_clusters["low_entitlements"] + filtered_clusters["zero_members"]
    if total_filtered > 0:
        logger.info(
            f"Filtered {total_filtered} clusters: "
            f"{filtered_clusters['low_entitlements']} low-entitlement, "
            f"{filtered_clusters['zero_members']} zero-member"
        )

    logger.info(
        f"Role building complete: {len(roles)} roles "
        f"(from {len(entitlement_clusters)} clusters, {total_filtered} filtered), "
        f"{summary['assigned_users']}/{summary['total_users']} users assigned"
    )

    return {
        "roles": roles,
        "birthright_role": birthright_role,
        "birthright_promotions": birthright_promotions,
        "merge_candidates": merge_candidates,
        "residuals": residuals,
        "summary": summary,
        "multi_cluster_info": multi_cluster_info,
        "naming_summary": naming_summary,
        "noise_entitlements": birthright_result.get("noise_entitlements", []),
        "filtered_clusters": filtered_clusters,
    }


# ============================================================================
# STEP 2-3: SPARSE PREVALENCE COMPUTATION
# ============================================================================

def _compute_prevalence_matrix(
        roles: List[Dict],
        matrix,  # CHANGE 2026-02-17: Now csr_matrix
        user_ids,  # CHANGE 2026-02-17: User IDs
        ent_ids,  # CHANGE 2026-02-17: Entitlement IDs
        excluded_ents: set,
) -> Dict[str, Any]:
    """
    Compute prevalence of every entitlement across every role using sparse
    matrix multiply (Option 3 from design spec).

    Operation: prevalence = (R.T @ M) / role_sizes
    Where R is a sparse (users x roles) membership matrix.

    Returns a dict so the result stays sparse (never densifies n_roles x n_ents):
        {
            "prevalence_csr": csr_matrix (n_roles x n_ents),  # values in [0, 1]
            "role_id_to_row": {role_id: row_index},
            "ent_id_to_col": {ent_id: col_index},
            "ent_ids": list[str],
        }
    Excluded entitlements (birthright + noise) have prevalence = 0 (dropped).
    """
    if not roles:
        return {
            "prevalence_csr": sp.csr_matrix((0, 0)),
            "role_id_to_row": {},
            "ent_id_to_col": {},
            "ent_ids": [],
        }

    all_users = list(user_ids)
    all_ents = list(ent_ids)
    user_idx = {uid: i for i, uid in enumerate(all_users)}
    n_users = len(all_users)
    n_roles = len(roles)

    # Build sparse membership matrix R (users x roles)
    row_indices = []
    col_indices = []
    for role_idx, role in enumerate(roles):
        for uid in role["members"]:
            if uid in user_idx:
                row_indices.append(user_idx[uid])
                col_indices.append(role_idx)

    R = sp.csr_matrix(
        (np.ones(len(row_indices), dtype=np.float64), (row_indices, col_indices)),
        shape=(n_users, n_roles),
    )

    M = matrix.astype(np.float64)

    # counts[r, e] = number of role-r members that hold entitlement e
    counts = (R.T @ M).tocsr()  # (n_roles x n_ents), stays sparse

    # Scale each row by 1/role_size to get prevalence — no densification
    role_sizes = np.asarray(R.sum(axis=0)).flatten()  # (n_roles,)
    safe_sizes = np.where(role_sizes > 0, role_sizes, 1.0)

    prevalence_csr = counts.copy()
    for r in range(n_roles):
        start = prevalence_csr.indptr[r]
        end = prevalence_csr.indptr[r + 1]
        if start < end:
            prevalence_csr.data[start:end] /= safe_sizes[r]

    # Zero out excluded entitlements by dropping their entries
    ent_id_to_col = {eid: i for i, eid in enumerate(all_ents)}
    excluded_cols = {ent_id_to_col[e] for e in excluded_ents if e in ent_id_to_col}
    if excluded_cols:
        mask = np.isin(prevalence_csr.indices, list(excluded_cols))
        prevalence_csr.data[mask] = 0.0
        prevalence_csr.eliminate_zeros()

    logger.info(
        f"Prevalence matrix: {n_roles} roles x {len(all_ents)} entitlements "
        f"({prevalence_csr.nnz} nonzero cells)"
    )

    return {
        "prevalence_csr": prevalence_csr,
        "role_id_to_row": {r["role_id"]: i for i, r in enumerate(roles)},
        "ent_id_to_col": ent_id_to_col,
        "ent_ids": all_ents,
    }


# ============================================================================
# STEP 4: CLASSIFY ENTITLEMENTS INTO TIERS
# ============================================================================

def _classify_entitlement_tiers(
        roles: List[Dict],
        prevalence_data: Dict[str, Any],
        config: Dict[str, Any],
) -> None:
    """
    Classify entitlements into core/common tiers for each role.
    Mutates role dicts in-place to add tier fields.

    Tiers:
    - Core: seed entitlements (always, even if low prevalence) +
            non-seed entitlements with prevalence >= prevalence_threshold
    - Common: non-seed entitlements with prevalence >= association_threshold
              and < prevalence_threshold
    - Low-prevalence seeds: seeds with prevalence < prevalence_threshold (flagged)
    """
    prevalence_threshold = config.get("entitlement_prevalence_threshold", 0.75)
    association_threshold = config.get("entitlement_association_threshold", 0.40)

    prevalence_csr = prevalence_data["prevalence_csr"]
    role_id_to_row = prevalence_data["role_id_to_row"]
    ent_ids = prevalence_data["ent_ids"]

    for role in roles:
        role_id = role["role_id"]
        seed_set = set(role["seed_entitlements"])

        if role_id not in role_id_to_row:
            role["core_entitlements"] = list(seed_set)
            role["common_entitlements"] = []
            role["entitlement_prevalence"] = {}
            role["low_prevalence_seeds"] = []
            continue

        row_idx = role_id_to_row[role_id]

        # Read nonzero entries for this role directly from CSR without densifying
        start = prevalence_csr.indptr[row_idx]
        end = prevalence_csr.indptr[row_idx + 1]
        col_indices = prevalence_csr.indices[start:end]
        prev_values = prevalence_csr.data[start:end]

        core = set(seed_set)
        common = []
        low_prevalence_seeds = []
        ent_prevalence_dict = {}

        for col_idx, prev in zip(col_indices, prev_values):
            if prev <= 0:
                continue
            ent_id = ent_ids[col_idx]
            if ent_id in seed_set:
                ent_prevalence_dict[ent_id] = round(float(prev), 4)
                if prev < prevalence_threshold:
                    low_prevalence_seeds.append(ent_id)
            elif prev >= prevalence_threshold:
                core.add(ent_id)
                ent_prevalence_dict[ent_id] = round(float(prev), 4)
            elif prev >= association_threshold:
                common.append(ent_id)
                ent_prevalence_dict[ent_id] = round(float(prev), 4)

        role["core_entitlements"] = sorted(core)
        role["common_entitlements"] = sorted(common)
        role["entitlement_prevalence"] = ent_prevalence_dict
        role["low_prevalence_seeds"] = sorted(low_prevalence_seeds)

    total_core = sum(len(r["core_entitlements"]) for r in roles)
    total_common = sum(len(r["common_entitlements"]) for r in roles)
    total_seeds = sum(len(r["seed_entitlements"]) for r in roles)
    promoted = total_core - total_seeds

    logger.info(
        f"Tier classification: {total_seeds} seed entitlements, "
        f"{promoted} promoted to core, {total_common} common"
    )


# ============================================================================
# STEP 5: BIRTHRIGHT PROMOTION SCAN
# ============================================================================

def _detect_birthright_promotions(
        roles: List[Dict],
        config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Detect entitlements that are core in a high percentage of all roles.
    These are candidates for escalation to the birthright role.

    Does NOT auto-move. Surfaces as a recommendation.

    Returns:
        List of {entitlement_id, core_in_roles, total_roles, pct}
    """
    threshold = config.get("birthright_promotion_threshold", 0.50)
    n_roles = len(roles)

    if n_roles == 0:
        return []

    core_counts: Dict[str, int] = {}
    for role in roles:
        for ent_id in role["core_entitlements"]:
            core_counts[ent_id] = core_counts.get(ent_id, 0) + 1

    promotions = []
    for ent_id, count in core_counts.items():
        pct = count / n_roles
        if pct > threshold:
            promotions.append({
                "entitlement_id": ent_id,
                "core_in_roles": count,
                "total_roles": n_roles,
                "pct": round(pct, 4),
            })

    promotions.sort(key=lambda x: -x["pct"])
    return promotions


# ============================================================================
# STEP 6: PAIRWISE ROLE OVERLAP
# ============================================================================

def _compute_role_overlap(
        roles: List[Dict],
        config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """
    Compute Jaccard similarity of core entitlement sets between all role pairs.
    Flag pairs above the merge similarity threshold.

    Returns:
        List of {role_a, role_b, jaccard, shared_core_count, shared_core}
    """
    threshold = config.get("role_merge_similarity_threshold", 0.70)
    n_roles = len(roles)

    if n_roles < 2:
        return []

    core_sets = [(r["role_id"], set(r["core_entitlements"])) for r in roles]

    merge_candidates = []
    for i in range(n_roles):
        id_a, set_a = core_sets[i]
        if not set_a:
            continue
        for j in range(i + 1, n_roles):
            id_b, set_b = core_sets[j]
            if not set_b:
                continue

            intersection = set_a & set_b
            union = set_a | set_b
            jaccard = len(intersection) / len(union) if union else 0.0

            if jaccard >= threshold:
                merge_candidates.append({
                    "role_a": id_a,
                    "role_b": id_b,
                    "jaccard": round(jaccard, 4),
                    "shared_core_count": len(intersection),
                    "shared_core": sorted(intersection),
                })

    merge_candidates.sort(key=lambda x: -x["jaccard"])
    return merge_candidates


# ============================================================================
# HR ATTRIBUTE SUMMARIZATION
# ============================================================================

def _summarize_hr_attributes(
        members: List[str],
        identities: pd.DataFrame,
        config: Dict[str, Any],
) -> Dict[str, List[Dict]]:
    """
    Summarize top attribute values for role members.

    Reads attribute columns from config["user_attributes"] list.
    Returns top 3 values per attribute with counts and percentages.
    """
    member_data = identities[identities['USR_ID'].isin(members)]
    if member_data.empty:
        return {}

    user_attributes = config.get("user_attributes", [])
    attr_columns = [attr["column"] for attr in user_attributes if "column" in attr]

    summary = {}

    for col in attr_columns:
        if col not in member_data.columns:
            continue

        values = member_data[col].dropna()
        values = values[values != ""]

        if values.empty:
            continue

        counts = values.value_counts().head(3)
        summary[col] = [
            {
                "value": str(v),
                "count": int(c),
                "pct": round(c / len(member_data), 4),
            }
            for v, c in counts.items()
        ]

    return summary


# ============================================================================
# ENTITLEMENT ENRICHMENT
# ============================================================================

def _enrich_entitlements(
        entitlement_ids: List[str],
        catalog: Optional[pd.DataFrame],
) -> List[Dict[str, str]]:
    """Enrich entitlement IDs with catalog metadata."""
    if catalog is not None and not catalog.empty:
        catalog_lookup = {}
        for idx, row in catalog.iterrows():
            try:
                catalog_lookup[row["namespaced_id"]] = {
                    "app_id": str(row["APP_ID"]),
                    "ent_name": str(row.get("ENT_NAME", row["ENT_ID"])),
                }
            except KeyError as e:
                logger.warning(f"Catalog row {idx} missing key: {e}")
                continue

        enriched = []
        for ent_id in entitlement_ids:
            detail = catalog_lookup.get(ent_id, {})
            enriched.append({
                "entitlement_id": ent_id,
                "app_id": detail.get("app_id", ent_id.split(":")[0]),
                "ent_name": detail.get("ent_name", ent_id.split(":")[-1]),
            })
        return enriched
    else:
        enriched = []
        for ent_id in entitlement_ids:
            parts = ent_id.split(":", 1)
            enriched.append({
                "entitlement_id": ent_id,
                "app_id": parts[0] if len(parts) == 2 else "",
                "ent_name": parts[1] if len(parts) == 2 else ent_id,
            })
        return enriched


# ============================================================================
# COVERAGE STATISTICS
# ============================================================================

def _compute_coverage_distribution(member_coverage: Dict[str, Dict]) -> Dict[str, int]:
    """Compute coverage distribution buckets."""
    if not member_coverage:
        return {}

    coverages = [data["coverage"] for data in member_coverage.values()]

    buckets = {
        "0.50-0.60": 0, "0.60-0.70": 0, "0.70-0.80": 0,
        "0.80-0.90": 0, "0.90-1.00": 0,
    }

    for cov in coverages:
        if 0.50 <= cov < 0.60:
            buckets["0.50-0.60"] += 1
        elif 0.60 <= cov < 0.70:
            buckets["0.60-0.70"] += 1
        elif 0.70 <= cov < 0.80:
            buckets["0.70-0.80"] += 1
        elif 0.80 <= cov < 0.90:
            buckets["0.80-0.90"] += 1
        elif 0.90 <= cov <= 1.00:
            buckets["0.90-1.00"] += 1

    return buckets


# ============================================================================
# BIRTHRIGHT ROLE
# ============================================================================

def _build_birthright_role(
        birthright_result: Dict[str, Any],
        catalog: Optional[pd.DataFrame],
) -> Dict[str, Any]:
    """Build birthright role from detection results."""
    birthright_ents = birthright_result["birthright_entitlements"]
    birthright_stats = birthright_result["birthright_stats"]

    entitlements_detail = _enrich_entitlements(birthright_ents, catalog)

    return {
        "role_id": "ROLE_BIRTHRIGHT",
        "role_name": "Organization-Wide Baseline Access",
        "entitlements": birthright_ents,
        "entitlement_count": len(birthright_ents),
        "stats": birthright_stats,
        "entitlements_detail": entitlements_detail,
    }


# ============================================================================
# RESIDUAL ACCESS COMPUTATION
# ============================================================================

def _compute_residuals_v2(
        matrix,  # CHANGE 2026-02-17: Now csr_matrix
        user_ids,  # CHANGE 2026-02-17: User IDs
        ent_ids,  # CHANGE 2026-02-17: Entitlement IDs
        roles: List[Dict],
        user_memberships: Dict[str, List[Dict]],
        birthright_entitlements: List[str],
) -> List[Dict[str, str]]:
    """
    Compute residual access (entitlements not covered by roles or birthright).

    CHANGE 2026-02-17: Updated to work with sparse matrix + indices.

    Uses core_entitlements (expanded) for coverage check.
    Multi-cluster membership means UNION of all assigned roles' core sets.
    """
    cluster_to_core = {}
    for role in roles:
        cluster_id = role["seed_cluster_id"]
        cluster_to_core[cluster_id] = set(role["core_entitlements"])

    birthright_set = set(birthright_entitlements)

    user_idx_map = {uid: i for i, uid in enumerate(user_ids)}
    ent_ids_array = np.array(ent_ids)

    residuals = []

    for user_id, memberships in user_memberships.items():
        covered_by_roles = set()
        for membership in memberships:
            cluster_id = membership["cluster_id"]
            core_ents = cluster_to_core.get(cluster_id, set())
            covered_by_roles.update(core_ents)

        total_covered = covered_by_roles | birthright_set

        if user_id in user_idx_map:
            u = user_idx_map[user_id]
            start, end = matrix.indptr[u], matrix.indptr[u + 1]
            user_ents = set(ent_ids_array[matrix.indices[start:end]])
        else:
            user_ents = set()

        residual_ents = user_ents - total_covered

        for ent_id in residual_ents:
            residuals.append({"USR_ID": user_id, "entitlement_id": ent_id})

    # Users with NO cluster membership
    # CHANGE 2026-02-17: Use user_ids parameter instead of matrix.index
    all_users = set(user_ids)
    assigned_users = set(user_memberships.keys())

    for user_id in (all_users - assigned_users):
        if user_id in user_idx_map:
            u = user_idx_map[user_id]
            start, end = matrix.indptr[u], matrix.indptr[u + 1]
            user_ents = set(ent_ids_array[matrix.indices[start:end]])
        else:
            user_ents = set()
        residual_ents = user_ents - birthright_set

        for ent_id in residual_ents:
            residuals.append({"USR_ID": user_id, "entitlement_id": ent_id})

    return residuals


# ============================================================================
# STATISTICS
# ============================================================================

def _compute_multi_cluster_stats(
        user_memberships: Dict[str, List[Dict]],
        entitlement_clusters: Dict[int, List[str]],
) -> Dict[str, Any]:
    """Compute statistics about multi-cluster membership."""
    cluster_counts = {}
    for memberships in user_memberships.values():
        count = len(memberships)
        cluster_counts[count] = cluster_counts.get(count, 0) + 1

    distribution = {
        f"{count}_cluster{'s' if count > 1 else ''}": users
        for count, users in sorted(cluster_counts.items())
    }

    combinations = {}
    for memberships in user_memberships.values():
        if len(memberships) <= 1:
            continue
        cluster_ids = tuple(sorted(m["cluster_id"] for m in memberships))
        combinations[cluster_ids] = combinations.get(cluster_ids, 0) + 1

    top_combinations = sorted(combinations.items(), key=lambda x: -x[1])[:10]

    most_common_combinations = [
        {
            "clusters": list(cluster_ids),
            "user_count": count,
            "description": f"Clusters {', '.join(str(c) for c in cluster_ids)}",
        }
        for cluster_ids, count in top_combinations
    ]

    return {
        "distribution": distribution,
        "most_common_combinations": most_common_combinations,
    }


def _compute_naming_summary(roles: List[Dict]) -> Dict[str, Any]:
    """Role naming summary. All roles are ROLE_NNN."""
    return {"total_roles": len(roles), "naming_method": "sequential"}


def _build_summary(
        roles: List[Dict],
        user_memberships: Dict[str, List[Dict]],
        birthright_result: Dict[str, Any],
        residuals: List[Dict],
        matrix,  # CHANGE 2026-02-17: Now csr_matrix
        user_ids,  # CHANGE 2026-02-17: User IDs for counting
) -> Dict[str, Any]:
    """
    Build overall summary statistics.

    CHANGE 2026-02-17: Updated to work with sparse matrix.
    """
    # CHANGE 2026-02-17: Use matrix.shape[0] instead of len(matrix)
    total_users = matrix.shape[0]
    assigned_users = len(user_memberships)

    multi_cluster_users = sum(
        1 for memberships in user_memberships.values()
        if len(memberships) > 1
    )

    total_memberships = sum(len(m) for m in user_memberships.values())
    avg_clusters = total_memberships / assigned_users if assigned_users > 0 else 0.0

    avg_coverage_values = [
        r["avg_coverage"] for r in roles if r.get("avg_coverage")
    ]
    overall_avg_coverage = np.mean(avg_coverage_values) if avg_coverage_values else 0.0

    total_core = sum(len(r.get("core_entitlements", [])) for r in roles)
    total_common = sum(len(r.get("common_entitlements", [])) for r in roles)
    total_seeds = sum(len(r.get("seed_entitlements", [])) for r in roles)

    return {
        "total_roles": len(roles),
        "total_users": total_users,
        "assigned_users": assigned_users,
        "unassigned_users": total_users - assigned_users,
        "multi_cluster_users": multi_cluster_users,
        "single_cluster_users": assigned_users - multi_cluster_users,
        "avg_clusters_per_user": round(avg_clusters, 2),
        "max_clusters_per_user": max(
            (len(m) for m in user_memberships.values()), default=0
        ),
        "avg_coverage": round(overall_avg_coverage, 4),
        "birthright_entitlements": len(birthright_result["birthright_entitlements"]),
        "noise_entitlements": len(birthright_result.get("noise_entitlements", [])),
        "residual_assignments": len(residuals),
        "total_seed_entitlements": total_seeds,
        "total_core_entitlements": total_core,
        "total_common_entitlements": total_common,
        "promoted_to_core": total_core - total_seeds,
    }
