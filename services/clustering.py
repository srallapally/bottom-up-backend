"""
Entitlement Clustering V2 - Leiden-Based Bottom-Up Role Mining
===============================================================

Implements entitlement-centric clustering using Leiden community detection.

Key differences from V1 (services/clustering.py):
- V1: User-centric (cluster users by entitlement similarity)
- V2: Entitlement-centric (cluster entitlements by user co-occurrence)
- V1: Hierarchical clustering (must specify k)
- V2: Leiden clustering (auto-determines k via modularity)
- V1: Single-cluster membership
- V2: Multi-cluster membership (users can belong to multiple roles)

Algorithm:
1. Transpose matrix: entitlements become rows, users become columns
2. Build sparse Jaccard similarity graph between entitlements
3. Run Leiden community detection to find entitlement clusters
4. Assign users to clusters based on coverage threshold
5. Drop small clusters (<min_role_size users)
6. Support both full clustering and incremental updates
"""

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, coo_matrix
from typing import Dict, List, Tuple, Optional, Any
import logging

# Leiden clustering imports
try:
    import igraph as ig
    import leidenalg

    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False
    logging.warning(
        "python-igraph or leidenalg not installed. "
        "Leiden clustering unavailable. Install with: "
        "pip install python-igraph leidenalg"
    )

logger = logging.getLogger(__name__)


# ============================================================================
# MAIN CLUSTERING FUNCTION
# ============================================================================

def cluster_entitlements_leiden(
        matrix,  # CHANGE 2026-02-17: Now accepts csr_matrix
        ent_ids,  # CHANGE 2026-02-17: Entitlement IDs (column labels)
        user_ids,  # CHANGE 2026-02-17: User IDs (row labels)
        leiden_min_similarity: float = 0.3,
        leiden_min_shared_users: int = 3,
        leiden_resolution: float = 1.0,
        leiden_random_seed: int = 42,
        min_entitlement_coverage: float = 0.5,
        min_absolute_overlap: int = 2,
        max_clusters_per_user: int = 5,
        min_role_size: int = 10,
        use_sparse: bool = True,
        graph_max_neighbors_per_node: int = 500,  # NEW: cap incident edges per entitlement node
        graph_max_entitlement_frequency_pct: float = 0.95,  # NEW: skip edges for entitlements held by >X% users
) -> Dict[str, Any]:
    """
    Bottom-up role mining via Leiden clustering on entitlement co-occurrence.

    CHANGE 2026-02-17: Now accepts sparse matrix + indices instead of DataFrame.

    Args:
        matrix: scipy.sparse.csr_matrix (users × entitlements) binary matrix
        ent_ids: array-like of entitlement IDs (column labels)
        user_ids: array-like of user IDs (row labels)
        leiden_min_similarity: Jaccard threshold for graph edges (0.2-0.7)
        leiden_min_shared_users: Min users sharing entitlements for edge (2+)
        leiden_resolution: Leiden granularity (0.5-2.0, higher=more clusters)
        leiden_random_seed: Random seed for reproducibility
        min_entitlement_coverage: User must have this % of cluster ents (0.3-0.8)
        min_absolute_overlap: User must match at least this many cluster ents (floor)
        max_clusters_per_user: Max roles per user (1-10)
        min_role_size: Drop clusters with fewer users (5+)
        use_sparse: Use sparse matrix operations (required for 50K+ users)
        graph_max_neighbors_per_node: Max incident edges per entitlement node (caps graph density)
        graph_max_entitlement_frequency_pct: Skip edge-building for entitlements held by >X% of users

    Returns:
        dict with:
            - entitlement_clusters: {cluster_id: [ent_ids]}
            - user_cluster_membership: {user_id: [{"cluster_id": X, "coverage": Y}]}
            - cluster_metadata: {cluster_id: stats}
            - leiden_stats: clustering quality metrics
            - n_clusters: final cluster count after filtering
            - unassigned_users: [user_ids] with no cluster coverage

    Raises:
        ImportError: If python-igraph or leidenalg not installed
        ValueError: If invalid parameters
    """
    if not LEIDEN_AVAILABLE:
        raise ImportError(
            "Leiden clustering requires python-igraph and leidenalg. "
            "Install with: pip install python-igraph leidenalg"
        )

    # Validate parameters
    _validate_clustering_params(
        leiden_min_similarity=leiden_min_similarity,
        leiden_min_shared_users=leiden_min_shared_users,
        leiden_resolution=leiden_resolution,
        min_entitlement_coverage=min_entitlement_coverage,
        min_absolute_overlap=min_absolute_overlap,
        max_clusters_per_user=max_clusters_per_user,
        min_role_size=min_role_size,
        graph_max_neighbors_per_node=graph_max_neighbors_per_node,
        graph_max_entitlement_frequency_pct=graph_max_entitlement_frequency_pct,
    )

    # CHANGE 2026-02-17: Calculate sparsity from sparse matrix
    logger.info(
        f"Starting Leiden clustering: {matrix.shape[0]} users, "
        f"{matrix.shape[1]} entitlements, sparsity={1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.4f}"
    )

    # Step 1: Transpose matrix (entitlements × users)
    logger.info("Step 1: Transposing matrix (entitlements become rows)")
    # CHANGE 2026-02-17: Matrix is already sparse, just transpose
    ent_matrix = matrix.T if use_sparse else matrix.T

    # Step 2: Build Jaccard similarity graph
    logger.info("Step 2: Building Jaccard similarity graph")
    edges, edge_weights = _build_jaccard_graph_sparse(
        ent_matrix=ent_matrix,
        min_similarity=leiden_min_similarity,
        min_shared_users=leiden_min_shared_users,
        entitlement_ids=ent_ids,  # CHANGE 2026-02-17: Use ent_ids parameter
        max_neighbors_per_node=graph_max_neighbors_per_node,
        max_entitlement_frequency_pct=graph_max_entitlement_frequency_pct,
    )


    # Diagnostics: summarize graph degree distribution (helps tune min_similarity/min_shared and sparsification knobs)
    # Note: degree stats are computed from the final edge list (after any gating / neighbor capping).
    degrees = np.zeros(len(ent_ids), dtype=np.int32)
    for a, b in edges:
        degrees[a] += 1
        degrees[b] += 1
    if len(ent_ids) > 0:
        isolated = int(np.sum(degrees == 0))
        # Percentiles over non-zero degrees are more informative; fall back to 0s if all isolated.
        nonzero = degrees[degrees > 0]
        if nonzero.size > 0:
            p50, p90, p99 = np.percentile(nonzero, [50, 90, 99])
            logger.info(
                f"Graph degree stats (nonzero): min={int(nonzero.min())} "
                f"p50={int(p50)} p90={int(p90)} p99={int(p99)} max={int(nonzero.max())}; "
                f"isolated_nodes={isolated}"
            )
        else:
            logger.info(f"Graph degree stats: all nodes isolated (isolated_nodes={isolated})")

    if len(edges) == 0:
        logger.warning("No edges in graph - data too sparse or thresholds too strict")
        # CHANGE 2026-02-17: Pass user_ids and ent_ids to _empty_clustering_result
        return _empty_clustering_result(matrix, user_ids, ent_ids)

    logger.info(f"Graph: {len(ent_ids)} nodes, {len(edges)} edges")  # CHANGE 2026-02-17

    # Step 3: Run Leiden clustering
    logger.info("Step 3: Running Leiden community detection")
    entitlement_clusters, leiden_stats = _run_leiden_clustering(
        n_entitlements=len(ent_ids),  # CHANGE 2026-02-17
        edges=edges,
        edge_weights=edge_weights,
        resolution=leiden_resolution,
        random_seed=leiden_random_seed,
        entitlement_ids=ent_ids,  # CHANGE 2026-02-17
    )

    logger.info(
        f"Leiden found {len(entitlement_clusters)} clusters "
        f"(cpm_quality={leiden_stats['cpm_quality']:.3f})"
    )

    # Step 4: Assign users to clusters (multi-membership)
    logger.info("Step 4: Assigning users to clusters (multi-membership)")
    user_memberships, cluster_user_counts = _assign_users_to_clusters(
        matrix=matrix,
        ent_ids=ent_ids,  # CHANGE 2026-02-17
        user_ids=user_ids,  # CHANGE 2026-02-17
        entitlement_clusters=entitlement_clusters,
        min_coverage=min_entitlement_coverage,
        min_absolute_overlap=min_absolute_overlap,
        max_clusters_per_user=max_clusters_per_user,
    )

    # Track assignment coverage before filtering (diagnostic)
    assigned_users_before_filter = set(user_memberships.keys())

    # Step 5: Drop small clusters
    logger.info("Step 5: Filtering small clusters")
    entitlement_clusters, user_memberships = _filter_small_clusters(
        entitlement_clusters=entitlement_clusters,
        user_memberships=user_memberships,
        cluster_user_counts=cluster_user_counts,
        min_role_size=min_role_size,
    )

    # Diagnostics: users that lost all memberships due to cluster filtering
    assigned_users_after_filter = set(user_memberships.keys())
    users_unassigned_by_filter = assigned_users_before_filter - assigned_users_after_filter
    if users_unassigned_by_filter:
        logger.info(
            f"Cluster filtering unassigned {len(users_unassigned_by_filter)} users "
            f"(due to clusters <{min_role_size} users)"
        )

    # Step 6: Compute metadata
    logger.info("Step 6: Computing cluster metadata")
    cluster_metadata = _compute_cluster_metadata(
        entitlement_clusters=entitlement_clusters,
        user_memberships=user_memberships,
        matrix=matrix,
    )

    # Find unassigned users
    # CHANGE 2026-02-17: Use user_ids parameter instead of matrix.index
    all_users = set(user_ids)
    assigned_users = set(user_memberships.keys())
    unassigned_users = list(all_users - assigned_users)

    logger.info(
        f"Clustering complete: {len(entitlement_clusters)} final clusters, "
        f"{len(assigned_users)}/{len(all_users)} users assigned"
    )

    return {
        "entitlement_clusters": entitlement_clusters,
        "user_cluster_membership": user_memberships,
        "cluster_metadata": cluster_metadata,
        "leiden_stats": leiden_stats,
        "n_clusters": len(entitlement_clusters),
        "unassigned_users": unassigned_users,
    }


# ============================================================================
# INCREMENTAL CLUSTERING (for daily updates)
# ============================================================================

def cluster_entitlements_incremental(
        matrix: csr_matrix,
        ent_ids: List[str],
        user_ids: List[str],
        previous_result: Dict[str, Any],
        config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Incremental clustering update (for daily reclustering).

    Only re-clusters if data change exceeds threshold, otherwise updates existing.

    NOTE:
        Incremental clustering is not implemented yet. This function currently
        falls back to full re-clustering.

    Args:
        matrix: Current user × entitlement matrix (csr_matrix)
        ent_ids: Entitlement IDs (column labels)
        user_ids: User IDs (row labels)
        previous_result: Previous clustering result
        config: Configuration dict with clustering parameters

    Returns:
        Updated clustering result (same schema as cluster_entitlements_leiden)
    """
    # Detect changes
    change_magnitude = _compute_data_change_magnitude(
        current_matrix=matrix,
        previous_result=previous_result,
    )

    logger.info(f"Data change magnitude: {change_magnitude:.2%}")

    # If change is small, try incremental update
    incremental_threshold = config.get("incremental_threshold", 0.05)
    if change_magnitude < incremental_threshold:
        logger.info(
            f"Change <{incremental_threshold:.0%}, attempting incremental update"
        )
        try:
            return _update_clusters_incrementally(
                matrix=matrix,
                previous_result=previous_result,
                config=config,
            )
        except Exception as e:
            logger.warning(f"Incremental update failed: {e}, falling back to full")

    # Fall back to full re-clustering
    logger.info("Running full re-clustering")
    return cluster_entitlements_leiden(
        matrix=matrix,
        ent_ids=ent_ids,
        user_ids=user_ids,
        leiden_min_similarity=config.get("leiden_min_similarity", 0.3),
        leiden_min_shared_users=config.get("leiden_min_shared_users", 3),
        leiden_resolution=config.get("leiden_resolution", 1.0),
        leiden_random_seed=config.get("leiden_random_seed", 42),
        min_entitlement_coverage=config.get("min_entitlement_coverage", 0.5),
        min_absolute_overlap=config.get("min_absolute_overlap", 2),
        max_clusters_per_user=config.get("max_clusters_per_user", 5),
        min_role_size=config.get("min_role_size", 10),
        use_sparse=config.get("use_sparse_matrices", True),
        graph_max_neighbors_per_node=config.get("graph_max_neighbors_per_node", 500),
        graph_max_entitlement_frequency_pct=config.get("graph_max_entitlement_frequency_pct", 0.95),
    )


# ============================================================================
# INTERNAL FUNCTIONS
# ============================================================================

def _validate_clustering_params(**params):
    """Validate clustering parameters."""
    sim = params["leiden_min_similarity"]
    if not 0.2 <= sim <= 0.7:
        raise ValueError(
            f"leiden_min_similarity={sim} must be in [0.2, 0.7]. "
            f"<0.2 creates giant cluster, >0.7 creates singletons"
        )

    min_shared = params["leiden_min_shared_users"]
    if min_shared < 2:
        raise ValueError(
            f"leiden_min_shared_users={min_shared} must be >=2"
        )

    coverage = params["min_entitlement_coverage"]
    if not 0.3 <= coverage <= 0.8:
        raise ValueError(
            f"min_entitlement_coverage={coverage} must be in [0.3, 0.8]. "
            f"<0.3 assigns users to too many roles, >0.8 leaves most unassigned"
        )

    resolution = params.get("leiden_resolution", 1.0)
    if not 0.1 <= resolution <= 5.0:
        raise ValueError(
            f"leiden_resolution={resolution} must be in [0.1, 5.0]."
        )

    min_abs = params.get("min_absolute_overlap", 1)
    if min_abs < 1:
        raise ValueError(
            f"min_absolute_overlap={min_abs} must be >=1"
        )

    max_per_user = params.get("max_clusters_per_user", 1)
    if max_per_user < 1:
        raise ValueError(
            f"max_clusters_per_user={max_per_user} must be >=1"
        )

    min_role_size = params.get("min_role_size", 1)
    if min_role_size < 1:
        raise ValueError(
            f"min_role_size={min_role_size} must be >=1"
        )

    # Graph sparsification controls
    max_neighbors = params.get("graph_max_neighbors_per_node", 0)
    if max_neighbors is not None and max_neighbors < 1:
        raise ValueError(
            f"graph_max_neighbors_per_node={max_neighbors} must be >=1"
        )

    max_freq = params.get("graph_max_entitlement_frequency_pct", 1.0)
    if max_freq is not None and not 0.0 < max_freq <= 1.0:
        raise ValueError(
            f"graph_max_entitlement_frequency_pct={max_freq} must be in (0, 1]."
        )


def _transpose_to_sparse(matrix: pd.DataFrame) -> csr_matrix:
    """
    Transpose matrix to sparse CSR format (entitlements × users).

    CHANGE 2026-02-17: DEPRECATED - No longer needed since matrix is already sparse.
    Kept for backward compatibility but should not be called.
    """
    # Convert to sparse if not already
    if isinstance(matrix, pd.DataFrame):
        sparse = csr_matrix(matrix.values.astype(np.int8))
    else:
        sparse = matrix

    # Transpose
    transposed = sparse.T
    return transposed


def _build_jaccard_graph_sparse(
        ent_matrix: csr_matrix,
        min_similarity: float,
        min_shared_users: int,
        entitlement_ids: List[str],
        max_neighbors_per_node: Optional[int] = None,
        max_entitlement_frequency_pct: Optional[float] = None,
) -> Tuple[List[Tuple[int, int]], List[float]]:
    """
    Build Jaccard similarity graph from sparse entitlement matrix.

    Only computes similarity for entitlement pairs that co-occur.
    This is O(non-zero pairs) instead of O(n²).

    Returns:
        edges: List of (ent_i, ent_j) tuples
        weights: List of Jaccard similarities
    """
    n_ents = ent_matrix.shape[0]

    # Compute overlap matrix: ent_matrix @ ent_matrix.T
    # overlap[i,j] = number of users with both ent_i and ent_j
    overlap = ent_matrix @ ent_matrix.T

    # Convert to COO for iteration
    overlap_coo = overlap.tocoo()

    # Pre-compute row sums (total users per entitlement)
    row_sums = np.asarray(ent_matrix.sum(axis=1)).flatten()

    # Optional: frequency gating to prevent ultra-common entitlements from creating
    # extremely dense overlap graphs. We only skip these entitlements for EDGE BUILDING;
    # they can still be handled elsewhere (e.g., birthright detection / prevalence tiers).
    excluded_nodes = None
    if max_entitlement_frequency_pct is not None:
        n_users = ent_matrix.shape[1]
        max_users = int(max_entitlement_frequency_pct * n_users)
        excluded_nodes = row_sums > max_users
        excluded_count = int(np.sum(excluded_nodes))
        if excluded_count > 0:
            logger.info(
                f"Graph frequency gating: excluding {excluded_count}/{n_ents} entitlements "
                f"with >{max_entitlement_frequency_pct:.0%} user coverage from edge building"
            )

    # Build edges
    edges = []
    weights = []

    for i, j, intersection in zip(overlap_coo.row, overlap_coo.col, overlap_coo.data):
        # Skip diagonal and upper triangle (undirected graph)
        if i >= j:
            continue

        # Skip entitlements excluded by frequency gating
        if excluded_nodes is not None and (excluded_nodes[i] or excluded_nodes[j]):
            continue

        # Filter by min shared users
        if intersection < min_shared_users:
            continue

        # Compute Jaccard similarity
        # Jaccard = intersection / union
        # union = |users_i| + |users_j| - |intersection|
        union = row_sums[i] + row_sums[j] - intersection
        jaccard = intersection / union if union > 0 else 0.0

        # Filter by min similarity
        if jaccard >= min_similarity:
            edges.append((int(i), int(j)))
            weights.append(float(jaccard))

    # Optional: cap the number of incident edges per node to keep the graph sparse.
    # We keep edges that are in the top-k (by weight) for EITHER endpoint.
    if max_neighbors_per_node is not None:
        k = int(max_neighbors_per_node)
        if k > 0 and len(edges) > 0:
            incident: Dict[int, List[Tuple[float, int]]] = {}
            for idx, ((a, b), w) in enumerate(zip(edges, weights)):
                incident.setdefault(a, []).append((w, idx))
                incident.setdefault(b, []).append((w, idx))

            keep = set()
            for node, items in incident.items():
                if len(items) <= k:
                    keep.update(idx for _, idx in items)
                    continue
                # Select top-k by weight (descending)
                items.sort(key=lambda t: t[0], reverse=True)
                keep.update(idx for _, idx in items[:k])

            if len(keep) < len(edges):
                edges = [edges[i] for i in sorted(keep)]
                weights = [weights[i] for i in sorted(keep)]
                logger.info(
                    f"Graph neighbor cap: retained {len(edges)} edges after capping "
                    f"to top-{k} incident edges per node"
                )

    return edges, weights


def _run_leiden_clustering(
        n_entitlements: int,
        edges: List[Tuple[int, int]],
        edge_weights: List[float],
        resolution: float,
        random_seed: int,
        entitlement_ids: List[str],
) -> Tuple[Dict[int, List[str]], Dict[str, Any]]:
    """
    Run Leiden community detection algorithm.

    Returns:
        entitlement_clusters: {cluster_id: [ent_ids]}
        leiden_stats: clustering quality metrics
    """
    # Build igraph
    g = ig.Graph()
    g.add_vertices(n_entitlements)
    g.add_edges(edges)
    g.es["weight"] = edge_weights

    # Run Leiden
    partition = leidenalg.find_partition(
        g,
        leidenalg.CPMVertexPartition,
        weights='weight',
        resolution_parameter=resolution,  # CPM supports this
        seed=random_seed,
        n_iterations=-1,
    )

    # Extract clusters
    entitlement_clusters = {}
    for ent_idx, cluster_id in enumerate(partition.membership):
        if cluster_id not in entitlement_clusters:
            entitlement_clusters[cluster_id] = []
        entitlement_clusters[cluster_id].append(entitlement_ids[ent_idx])

    # Re-number clusters starting from 1 (not 0)
    old_to_new = {old: new for new, old in enumerate(sorted(entitlement_clusters.keys()), start=1)}
    entitlement_clusters = {
        old_to_new[old_id]: ents
        for old_id, ents in entitlement_clusters.items()
    }

    # Compute stats
    leiden_stats = {
        "cpm_quality": float(partition.quality()),
        "resolution": resolution,
        "initial_clusters": len(partition),
        "graph_nodes": n_entitlements,
        "graph_edges": len(edges),
        "avg_degree": 2 * len(edges) / n_entitlements if n_entitlements > 0 else 0,
        "isolated_nodes": n_entitlements - len(set(node for edge in edges for node in edge)),
    }

    return entitlement_clusters, leiden_stats


def _assign_users_to_clusters(
        matrix,  # CHANGE 2026-02-17: Now csr_matrix
        ent_ids,  # CHANGE 2026-02-17: Entitlement IDs
        user_ids,  # CHANGE 2026-02-17: User IDs
        entitlement_clusters: Dict[int, List[str]],
        min_coverage: float,
        min_absolute_overlap: int,
        max_clusters_per_user: int,
) -> Tuple[Dict[str, List[Dict]], Dict[int, int]]:
    """
    Assign users to clusters based on coverage threshold.

    CHANGE 2026-02-17: Updated to work with sparse matrix + indices.

    Users can belong to multiple clusters (multi-membership).
    Both conditions must be satisfied: coverage >= min_coverage AND
    absolute overlap >= min_absolute_overlap. The floor prevents
    single-entitlement matches from passing the percentage threshold
    on small clusters.

    Vectorized implementation: replaces O(n_users x n_clusters) Python loop
    with sparse matrix multiplication O(nnz), where nnz is the number of
    non-zero entries in the user-entitlement matrix (~460K for 50K users).

    Returns:
        user_memberships: {user_id: [{"cluster_id": X, "coverage": Y, ...}]}
        cluster_user_counts: {cluster_id: user_count}
    """
    cluster_ids = sorted(entitlement_clusters.keys())
    # CHANGE 2026-02-17: Use ent_ids parameter instead of matrix.columns
    ent_index = {ent: i for i, ent in enumerate(ent_ids)}
    n_ents = len(ent_ids)
    n_clusters = len(cluster_ids)

    # Build cluster indicator matrix: (n_clusters x n_ents), binary
    # Row c has 1s at positions of cluster c's entitlements
    cluster_sizes = np.zeros(n_clusters, dtype=np.int32)
    rows, cols = [], []
    for c_pos, cid in enumerate(cluster_ids):
        for ent in entitlement_clusters[cid]:
            if ent in ent_index:
                rows.append(c_pos)
                cols.append(ent_index[ent])
        # cluster_sizes is computed below using bincount to avoid per-cluster list allocation
    cluster_matrix = csr_matrix(
        (np.ones(len(rows), dtype=np.int32), (rows, cols)),
        shape=(n_clusters, n_ents),
    )

    # Compute cluster sizes without allocating a list per cluster
    if len(rows) > 0:
        cluster_sizes[:] = np.bincount(
            np.asarray(rows, dtype=np.int32),
            minlength=n_clusters,
        ).astype(np.int32)

    # Sparse user matrix: (n_users x n_ents)
    # CHANGE 2026-02-17: Matrix is already sparse, no need for conversion
    user_matrix = matrix.astype(np.int32)

    # overlap_sparse[u, c] = number of cluster c entitlements user u holds
    # Kept sparse to avoid materializing n_users × n_clusters dense matrix
    overlap_sparse = (user_matrix @ cluster_matrix.T).tocsr()

    # Avoid division by zero for empty clusters (shouldn't occur post-filtering)
    safe_sizes = np.where(cluster_sizes > 0, cluster_sizes, 1)

    # Build output dicts
    # CHANGE 2026-02-17: Use user_ids parameter instead of matrix.index
    user_memberships = {}
    cluster_user_counts = {cid: 0 for cid in cluster_ids}

    for u_pos, user_id in enumerate(user_ids):
        row_start = overlap_sparse.indptr[u_pos]
        row_end = overlap_sparse.indptr[u_pos + 1]
        if row_start == row_end:
            continue  # user has no overlap with any cluster

        c_positions = overlap_sparse.indices[row_start:row_end]   # cluster column positions
        overlap_vals = overlap_sparse.data[row_start:row_end].astype(np.float64)
        coverage_vals = overlap_vals / safe_sizes[c_positions]

        # Apply both thresholds
        mask = (coverage_vals >= min_coverage) & (overlap_vals >= min_absolute_overlap)
        qualifying_local = np.where(mask)[0]
        if len(qualifying_local) == 0:
            continue

        # Sort by coverage descending; break ties by overlap count descending.
        # This reduces bias toward tiny clusters that can yield high coverage with small absolute overlap.
        idxs = qualifying_local
        order = np.lexsort((-overlap_vals[idxs], -coverage_vals[idxs]))  # primary: coverage, secondary: overlap
        qualifying_local = idxs[order]
        qualifying_local = qualifying_local[:max_clusters_per_user]

        memberships = []
        for idx in qualifying_local:
            c_pos = c_positions[idx]
            cid = cluster_ids[c_pos]
            memberships.append({
                "cluster_id": cid,
                "coverage": round(float(coverage_vals[idx]), 4),
                "count": int(overlap_vals[idx]),
                "total": int(cluster_sizes[c_pos]),
            })
            cluster_user_counts[cid] += 1

        user_memberships[user_id] = memberships

    return user_memberships, cluster_user_counts


def _filter_small_clusters(
        entitlement_clusters: Dict[int, List[str]],
        user_memberships: Dict[str, List[Dict]],
        cluster_user_counts: Dict[int, int],
        min_role_size: int,
) -> Tuple[Dict[int, List[str]], Dict[str, List[Dict]]]:
    """
    Drop clusters with fewer than min_role_size users.

    Returns:
        Filtered entitlement_clusters and user_memberships
    """
    # Find valid clusters
    valid_clusters = {
        cid for cid, count in cluster_user_counts.items()
        if count >= min_role_size
    }

    if not valid_clusters:
        logger.warning("All clusters dropped (too small), returning empty")
        return {}, {}

    dropped_count = len(entitlement_clusters) - len(valid_clusters)
    if dropped_count > 0:
        logger.info(f"Dropped {dropped_count} small clusters (<{min_role_size} users)")

    # Filter entitlement clusters
    filtered_clusters = {
        cid: ents for cid, ents in entitlement_clusters.items()
        if cid in valid_clusters
    }

    # Filter user memberships
    filtered_memberships = {}
    for user_id, memberships in user_memberships.items():
        valid_memberships = [
            m for m in memberships
            if m["cluster_id"] in valid_clusters
        ]
        if valid_memberships:
            filtered_memberships[user_id] = valid_memberships

    return filtered_clusters, filtered_memberships


def _compute_cluster_metadata(
        entitlement_clusters: Dict[int, List[str]],
        user_memberships: Dict[str, List[Dict]],
        matrix,
) -> Dict[int, Dict[str, Any]]:
    """Compute per-cluster statistics."""
    metadata = {}

    for cluster_id, cluster_ents in entitlement_clusters.items():
        # Find users in this cluster
        users_in_cluster = [
            user_id for user_id, memberships in user_memberships.items()
            if any(m["cluster_id"] == cluster_id for m in memberships)
        ]

        # Get coverage stats
        coverages = [
            m["coverage"] for user_id in users_in_cluster
            for m in user_memberships[user_id]
            if m["cluster_id"] == cluster_id
        ]

        metadata[cluster_id] = {
            "entitlement_count": len(cluster_ents),
            "user_count": len(users_in_cluster),
            "avg_coverage": round(np.mean(coverages), 4) if coverages else 0.0,
            "min_coverage": round(np.min(coverages), 4) if coverages else 0.0,
            "max_coverage": round(np.max(coverages), 4) if coverages else 0.0,
        }

    return metadata


def _empty_clustering_result(matrix, user_ids, ent_ids) -> Dict[str, Any]:
    """
    Return empty result when clustering fails.

    CHANGE 2026-02-17: Updated to work with sparse matrix + indices.
    """
    return {
        "entitlement_clusters": {},
        "user_cluster_membership": {},
        "cluster_metadata": {},
        "leiden_stats": {
            "cpm_quality": 0.0,
            "initial_clusters": 0,
            "graph_nodes": len(ent_ids),  # CHANGE 2026-02-17
            "graph_edges": 0,
        },
        "n_clusters": 0,
        "unassigned_users": list(user_ids),  # CHANGE 2026-02-17
    }


def _compute_data_change_magnitude(
        current_matrix: csr_matrix,
        previous_result: Dict[str, Any],
) -> float:
    """
    Compute magnitude of data change since last clustering.

    Returns:
        Float in [0, 1] representing % change
    """
    # TODO: Implement proper change detection
    # For now, return 1.0 (always do full clustering)
    return 1.0


def _update_clusters_incrementally(
        matrix: csr_matrix,
        previous_result: Dict[str, Any],
        config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Update clusters incrementally (optimization for daily updates).

    TODO: Implement incremental update logic.
    For now, raises NotImplementedError to fall back to full clustering.
    """
    raise NotImplementedError("Incremental clustering not yet implemented")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example with synthetic data
    np.random.seed(42)

    # Create synthetic user × entitlement matrix
    # 1000 users, 200 entitlements, ~99% sparse
    n_users = 1000
    n_ents = 200
    sparsity = 0.99

    # Generate random sparse matrix
    n_assignments = int(n_users * n_ents * (1 - sparsity))
    user_ids = [f"USR_{i:04d}" for i in range(n_users)]
    ent_ids = [f"APP_{i // 20}:ENT_{i % 20}" for i in range(n_ents)]

    # Random assignments
    users_idx = np.random.choice(n_users, size=n_assignments, replace=True)
    ents_idx = np.random.choice(n_ents, size=n_assignments, replace=True)

    # Build sparse CSR matrix directly
    data = np.ones(len(users_idx), dtype=np.int8)
    matrix = csr_matrix(
        (data, (users_idx, ents_idx)),
        shape=(n_users, n_ents),
        dtype=np.int8,
    )

    print(f"Synthetic matrix: {matrix.shape[0]} users × {matrix.shape[1]} entitlements")
    print(f"Sparsity: {1 - (matrix.nnz / (n_users * n_ents)):.2%}")
    print(f"Avg entitlements/user: {float(matrix.sum(axis=1).mean()):.1f}")

    # Run clustering
    if LEIDEN_AVAILABLE:
        result = cluster_entitlements_leiden(
            matrix=matrix,
            ent_ids=ent_ids,
            user_ids=user_ids,
            leiden_min_similarity=0.3,
            leiden_min_shared_users=3,
            leiden_resolution=1.0,
            min_entitlement_coverage=0.5,
            max_clusters_per_user=5,
            min_role_size=10,
            graph_max_neighbors_per_node=500,
            graph_max_entitlement_frequency_pct=0.95,
        )

        print(f"\nClustering results:")
        print(f"  - Clusters: {result['n_clusters']}")
        print(f"  - CPM quality: {result['leiden_stats']['cpm_quality']:.3f}")
        print(f"  - Assigned users: {len(result['user_cluster_membership'])}/{n_users}")
        print(f"  - Unassigned users: {len(result['unassigned_users'])}")

        # Show cluster sizes
        print(f"\nCluster sizes:")
        for cid, meta in sorted(result['cluster_metadata'].items()):
            print(
                f"  Cluster {cid}: {meta['user_count']} users, {meta['entitlement_count']} ents, avg_coverage={meta['avg_coverage']:.2%}")
    else:
        print("\nLeiden not available - install with: pip install python-igraph leidenalg")
