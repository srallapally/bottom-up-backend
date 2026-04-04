# services/hybrid_miner.py
"""
Hybrid Miner — Sparse Co-occurrence Role Discovery for Leiden Residuals
========================================================================

Runs AFTER Leiden role building (Step 6) to discover supplementary roles
from the residual population — users and entitlements not covered by
Leiden clusters.

Algorithm:
1. Build a sparse sub-matrix of residual users × residual entitlements
2. Compute entitlement co-occurrence via sparse matrix multiply (E.T @ E)
3. Find frequent pairs (co-occurrence >= min_users), then greedily grow
   each pair into the largest clique (fully-connected subgraph) where
   every pair maintains the min_co_occurrence user support threshold
4. Deduplicate: keep only maximal cliques (remove strict subsets)
5. Greedy exclusive selection: rank by (ent_count × member_count),
   commit the best role, REMOVE its entitlements from all remaining
   candidates, repeat. This enforces zero entitlement overlap across
   hybrid roles.

Output: list of role dicts compatible with the existing roles schema
(role_id, core_entitlements, members, member_count, entitlement_prevalence).

These roles are merged into the roles list and passed through to the
confidence scorer. They use role_ids starting from HYBRID_001.

Performance: The sparse E.T @ E multiply is O(nnz^2/n) and completes in
<2s for the OC dataset. Clique growth is bounded by the pair count and
adjacency density. No dense matrix is ever constructed.
"""

import numpy as np
import time
import logging
from collections import defaultdict
from typing import Dict, List, Any, Optional, Set
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)


def discover_hybrid_roles(
        matrix,                # csr_matrix (all users × all ents), original full matrix
        user_ids,              # array of user ID strings (row labels for matrix)
        ent_ids,               # array of entitlement ID strings (col labels for matrix)
        leiden_roles: List[Dict],           # existing Leiden roles from build_roles
        user_memberships: Dict[str, Any],   # cluster_result["user_cluster_membership"]
        birthright_entitlements: List[str],
        tiered_birthright_roles: Optional[List[Dict]] = None,
        min_co_occurrence: int = 25,        # minimum users sharing an ent pair
        min_co_occurrence_growth: Optional[int] = None,  # relaxed threshold for clique extension (default: min_co_occurrence // 2, floor 3)
        min_role_users: int = 3,            # minimum users to form a hybrid role
        min_entitlements_per_role: int = 3,  # minimum 3 ents per role
) -> Dict[str, Any]:
    """
    Discover supplementary roles from Leiden residual population.

    Args:
        matrix: full csr_matrix (users × ents) — NOT the filtered matrix
        user_ids: row labels
        ent_ids: column labels
        leiden_roles: roles from build_roles (to determine covered ents/users)
        user_memberships: cluster_result["user_cluster_membership"]
        birthright_entitlements: universal birthright ent IDs
        tiered_birthright_roles: tiered birthright role dicts
        min_co_occurrence: minimum co-occurrence count for a pair
        min_role_users: minimum users to keep a hybrid role
        min_entitlements_per_role: minimum entitlements per role (always >=2)

    Returns:
        dict with:
            - hybrid_roles: list of role dicts compatible with Leiden roles
            - hybrid_stats: summary statistics
    """
    _t_start = time.monotonic()

    # Resolve growth threshold: used for clique extension (Step 3).
    # Pair discovery (Step 2) still uses the strict min_co_occurrence.
    if min_co_occurrence_growth is None:
        min_co_occurrence_growth = max(min_co_occurrence // 2, 3)
    logger.info(
        "hybrid_miner: min_co_occurrence=%d, min_co_occurrence_growth=%d",
        min_co_occurrence, min_co_occurrence_growth,
    )

    user_ids_arr = np.array(user_ids)
    ent_ids_arr = np.array(ent_ids)
    user_idx_map = {uid: i for i, uid in enumerate(user_ids_arr)}
    ent_idx_map = {eid: i for i, eid in enumerate(ent_ids_arr)}

    # ---- Step 1: Identify residual population ----
    # Residual users = users NOT assigned to any Leiden cluster
    assigned_users = set(user_memberships.keys())
    all_users = set(user_ids_arr)
    residual_user_ids = sorted(all_users - assigned_users)

    if not residual_user_ids:
        logger.info("hybrid_miner: no residual users, skipping")
        return {"hybrid_roles": [], "hybrid_stats": _empty_stats()}

    # Entitlements already explained by Leiden core, birthright, or tiered BR
    covered_ents: Set[str] = set(birthright_entitlements)
    for role in leiden_roles:
        covered_ents.update(role.get("core_entitlements", []))
    for tbr in (tiered_birthright_roles or []):
        covered_ents.update(tbr["entitlements"])

    # Build residual sub-matrix: residual users × all non-covered ents
    residual_row_indices = np.array([user_idx_map[u] for u in residual_user_ids], dtype=np.int32)

    # Identify candidate entitlement columns (not covered, and held by >= min_co_occurrence residual users)
    sub_matrix = matrix[residual_row_indices, :]
    ent_counts = np.asarray(sub_matrix.sum(axis=0)).flatten()

    candidate_ent_mask = np.ones(len(ent_ids_arr), dtype=bool)
    for eid in covered_ents:
        idx = ent_idx_map.get(eid)
        if idx is not None:
            candidate_ent_mask[idx] = False
    candidate_ent_mask &= (ent_counts >= min_co_occurrence)

    candidate_col_indices = np.where(candidate_ent_mask)[0]
    if len(candidate_col_indices) < min_entitlements_per_role:
        logger.info(
            "hybrid_miner: only %d candidate entitlements, need >= %d. Skipping.",
            len(candidate_col_indices), min_entitlements_per_role,
        )
        return {"hybrid_roles": [], "hybrid_stats": _empty_stats()}

    # Slice to residual sub-matrix (residual_users × candidate_ents)
    res_matrix = sub_matrix[:, candidate_col_indices]
    res_ent_ids = ent_ids_arr[candidate_col_indices]
    res_ent_idx_map = {eid: i for i, eid in enumerate(res_ent_ids)}

    logger.info(
        "hybrid_miner Step 1: %d residual users, %d candidate entitlements, nnz=%d",
        len(residual_user_ids), len(res_ent_ids), res_matrix.nnz,
    )

    # ---- Step 2: Sparse co-occurrence matrix (E.T @ E) ----
    _t = time.monotonic()
    ent_matrix = res_matrix.T.tocsr()
    cooccur = (ent_matrix @ ent_matrix.T).tocoo()

    pairs = []
    for i, j, count in zip(cooccur.row, cooccur.col, cooccur.data):
        if i < j and count >= min_co_occurrence:
            pairs.append((i, j, int(count)))

    logger.info(
        "hybrid_miner Step 2: %d frequent pairs (>=%d users) in %.0fms",
        len(pairs), min_co_occurrence, (time.monotonic() - _t) * 1000,
    )

    if not pairs:
        return {"hybrid_roles": [], "hybrid_stats": _empty_stats()}

    # ---- Step 3: Grow maximal cliques from pairs ----
    # Instead of stopping at triples, greedily extend each pair into the
    # largest clique (fully-connected subgraph) where every entitlement
    # pair has co-occurrence >= min_co_occurrence users.
    _t = time.monotonic()
    adj = defaultdict(set)
    for i, j, _ in pairs:
        adj[i].add(j)
        adj[j].add(i)

    # Pre-compute user sets per entitlement for intersection
    ent_user_sets = {}
    for idx in range(len(res_ent_ids)):
        ent_user_sets[idx] = set(ent_matrix[idx].indices)

    pair_set = {(min(i, j), max(i, j)) for i, j, _ in pairs}

    def _grow_clique(seed: frozenset) -> frozenset:
        """Greedily extend a seed clique by adding entitlements that are
        adjacent to ALL current members in the pair graph AND whose
        intersection of user sets with the clique has >= min_co_occurrence_growth.
        Note: pair discovery uses the strict min_co_occurrence; extension uses
        the relaxed min_co_occurrence_growth to allow 3+ ent cliques on sparse
        residual data where multi-way intersections shrink rapidly."""
        clique = set(seed)
        # Candidates = neighbors of every node in the clique
        candidates = set.intersection(*(adj[n] for n in clique)) - clique
        # Sort candidates by user set size descending (prefer high-frequency first)
        candidates = sorted(candidates, key=lambda c: len(ent_user_sets[c]), reverse=True)
        for c in candidates:
            # Check: c must form a frequent pair with every member of current clique
            if all((min(c, m), max(c, m)) in pair_set for m in clique):
                # Check user support: intersection of all user sets including c
                clique_users = set.intersection(*(ent_user_sets[m] for m in clique))
                extended_users = clique_users & ent_user_sets[c]
                if len(extended_users) >= min_co_occurrence_growth:
                    clique.add(c)
        return frozenset(clique)

    # Grow from every pair
    seen_cliques: Set[frozenset] = set()
    grown_cliques = []
    for i, j, _ in pairs:
        seed = frozenset([i, j])
        clique = _grow_clique(seed)
        if clique not in seen_cliques:
            seen_cliques.add(clique)
            clique_users = set.intersection(*(ent_user_sets[e] for e in clique))
            grown_cliques.append((clique, len(clique_users)))

    logger.info(
        "hybrid_miner Step 3: %d unique cliques grown from %d pairs (max size %d) in %.0fms",
        len(grown_cliques), len(pairs),
        max((len(c) for c, _ in grown_cliques), default=0),
        (time.monotonic() - _t) * 1000,
    )

    # ---- Step 4: Keep only maximal itemsets ----
    # Remove any clique that is a strict subset of another
    _t = time.monotonic()
    # Sort largest first for efficient subset checking
    grown_cliques.sort(key=lambda x: len(x[0]), reverse=True)
    maximal = []
    maximal_sets = []
    for ent_set, user_count in grown_cliques:
        if len(ent_set) < min_entitlements_per_role:
            continue
        if any(ent_set < ms for ms in maximal_sets):
            continue
        maximal.append((ent_set, user_count))
        maximal_sets.append(ent_set)

    logger.info(
        "hybrid_miner Step 4: %d cliques -> %d maximal (>=%d ents)",
        len(grown_cliques), len(maximal), min_entitlements_per_role,
    )

    # ---- Step 5: Greedy exclusive role selection ----
    # Sort by (entitlement_count × member_count) descending — prefers roles
    # that explain the most assignment pairs
    maximal.sort(key=lambda x: len(x[0]) * x[1], reverse=True)

    consumed_ents: Set[int] = set()   # entitlement indices already committed
    hybrid_roles = []
    role_counter = 0

    for ent_set, _ in maximal:
        # Remove entitlements already consumed by earlier roles
        remaining_ents = ent_set - consumed_ents
        if len(remaining_ents) < min_entitlements_per_role:
            continue

        # Compute exact user set for the remaining entitlements
        role_user_indices = set.intersection(*(ent_user_sets[e] for e in remaining_ents))
        if len(role_user_indices) < min_role_users:
            continue

        role_counter += 1
        role_id = f"HYBRID_{role_counter:03d}"

        # Commit these entitlements — no other role can reuse them
        consumed_ents.update(remaining_ents)

        # Map back to actual user IDs and entitlement IDs
        members = [residual_user_ids[idx] for idx in sorted(role_user_indices)]
        core_ent_ids = sorted([res_ent_ids[e] for e in remaining_ents])

        # Compute entitlement prevalence within this role
        ent_prevalence = {}
        for eid in core_ent_ids:
            local_idx = res_ent_idx_map[eid]
            holders = len(ent_user_sets[local_idx] & role_user_indices)
            ent_prevalence[eid] = round(holders / len(role_user_indices), 4)

        # Build member_coverage
        member_coverage = {}
        n_core = len(core_ent_ids)
        core_col_indices = [res_ent_idx_map[eid] for eid in core_ent_ids]
        for u_local_idx in sorted(role_user_indices):
            uid = residual_user_ids[u_local_idx]
            has_count = sum(
                1 for c in core_col_indices
                if res_matrix[u_local_idx, c] != 0
            )
            member_coverage[uid] = {
                "coverage": round(has_count / n_core, 4) if n_core > 0 else 0.0,
                "has_count": has_count,
                "total_count": n_core,
            }

        hybrid_roles.append({
            "role_id": role_id,
            "role_name": role_id,
            "seed_cluster_id": None,
            "source": "hybrid_cooccurrence",
            "core_entitlements": core_ent_ids,
            "common_entitlements": [],
            "entitlements": core_ent_ids,
            "entitlement_count": len(core_ent_ids),
            "entitlement_prevalence": ent_prevalence,
            "members": members,
            "member_count": len(members),
            "member_coverage": member_coverage,
            "avg_coverage": round(
                np.mean([mc["coverage"] for mc in member_coverage.values()]), 4
            ),
            "min_coverage": round(
                min(mc["coverage"] for mc in member_coverage.values()), 4
            ),
            "max_coverage": round(
                max(mc["coverage"] for mc in member_coverage.values()), 4
            ),
            "coverage_distribution": {},
            "hr_summary": {},
            "status": "draft",
        })

    # Compute coverage stats
    total_hybrid_users = set()
    for role in hybrid_roles:
        total_hybrid_users.update(role["members"])

    elapsed_ms = (time.monotonic() - _t_start) * 1000
    stats = {
        "residual_users_input": len(residual_user_ids),
        "candidate_entitlements": len(res_ent_ids),
        "frequent_pairs": len(pairs),
        "unique_cliques_grown": len(grown_cliques),
        "maximal_itemsets": len(maximal),
        "hybrid_roles_created": len(hybrid_roles),
        "consumed_entitlements": len(consumed_ents),
        "hybrid_users_covered": len(total_hybrid_users),
        "hybrid_coverage_pct": round(
            len(total_hybrid_users) / len(residual_user_ids), 4
        ) if residual_user_ids else 0.0,
        "elapsed_ms": round(elapsed_ms),
    }

    logger.info(
        "hybrid_miner complete: %d roles covering %d/%d residual users (%.1f%%) in %.0fms",
        len(hybrid_roles), len(total_hybrid_users), len(residual_user_ids),
        stats["hybrid_coverage_pct"] * 100, elapsed_ms,
    )

    return {
        "hybrid_roles": hybrid_roles,
        "hybrid_stats": stats,
    }


def merge_hybrid_into_cluster_result(
        cluster_result: Dict[str, Any],
        hybrid_roles: List[Dict],
) -> Dict[str, Any]:
    """
    Merge hybrid role members into cluster_result["user_cluster_membership"]
    so the confidence scorer can look up hybrid users as assigned.

    Hybrid roles get synthetic cluster_ids starting from max_existing + 1000
    to avoid collisions with Leiden cluster IDs.

    Mutates cluster_result in place and returns it.
    """
    if not hybrid_roles:
        return cluster_result

    user_memberships = cluster_result["user_cluster_membership"]
    ent_clusters = cluster_result["entitlement_clusters"]

    # Find max existing cluster_id to avoid collisions
    existing_ids = list(ent_clusters.keys())
    base_id = (max(existing_ids) + 1000) if existing_ids else 1000

    for i, role in enumerate(hybrid_roles):
        synthetic_cluster_id = base_id + i
        core_ents = role["core_entitlements"]

        # Add to entitlement_clusters
        ent_clusters[synthetic_cluster_id] = core_ents

        # Add members to user_cluster_membership
        for uid in role["members"]:
            mc = role["member_coverage"].get(uid, {})
            membership_entry = {
                "cluster_id": synthetic_cluster_id,
                "coverage": mc.get("coverage", 1.0),
                "count": mc.get("has_count", len(core_ents)),
                "total": mc.get("total_count", len(core_ents)),
            }
            if uid not in user_memberships:
                user_memberships[uid] = []
            user_memberships[uid].append(membership_entry)

    cluster_result["n_clusters"] = len(ent_clusters)

    logger.info(
        "Merged %d hybrid roles into cluster_result (synthetic IDs %d-%d)",
        len(hybrid_roles), base_id, base_id + len(hybrid_roles) - 1,
    )

    return cluster_result


def _empty_stats():
    return {
        "residual_users_input": 0,
        "candidate_entitlements": 0,
        "frequent_pairs": 0,
        "unique_cliques_grown": 0,
        "maximal_itemsets": 0,
        "hybrid_roles_created": 0,
        "consumed_entitlements": 0,
        "hybrid_users_covered": 0,
        "hybrid_coverage_pct": 0.0,
        "elapsed_ms": 0,
    }