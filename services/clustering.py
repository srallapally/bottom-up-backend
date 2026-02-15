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
        matrix: pd.DataFrame,
        leiden_min_similarity: float = 0.3,
        leiden_min_shared_users: int = 3,
        leiden_resolution: float = 1.0,
        leiden_random_seed: int = 42,
        min_entitlement_coverage: float = 0.5,
        max_clusters_per_user: int = 5,
        min_role_size: int = 10,
        use_sparse: bool = True,
) -> Dict[str, Any]:
    """
    Bottom-up role mining via Leiden clustering on entitlement co-occurrence.

    Args:
        matrix: User × Entitlement binary matrix (users=rows, ents=cols)
        leiden_min_similarity: Jaccard threshold for graph edges (0.2-0.7)
        leiden_min_shared_users: Min users sharing entitlements for edge (2+)
        leiden_resolution: Leiden granularity (0.5-2.0, higher=more clusters)
        leiden_random_seed: Random seed for reproducibility
        min_entitlement_coverage: User must have this % of cluster ents (0.3-0.8)
        max_clusters_per_user: Max roles per user (1-10)
        min_role_size: Drop clusters with fewer users (5+)
        use_sparse: Use sparse matrix operations (required for 50K+ users)

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
        max_clusters_per_user=max_clusters_per_user,
        min_role_size=min_role_size,
    )

    logger.info(
        f"Starting Leiden clustering: {matrix.shape[0]} users, "
        f"{matrix.shape[1]} entitlements, sparsity={1 - matrix.values.mean():.4f}"
    )

    # Step 1: Transpose matrix (entitlements × users)
    logger.info("Step 1: Transposing matrix (entitlements become rows)")
    ent_matrix = _transpose_to_sparse(matrix) if use_sparse else matrix.T

    # Step 2: Build Jaccard similarity graph
    logger.info("Step 2: Building Jaccard similarity graph")
    edges, edge_weights = _build_jaccard_graph_sparse(
        ent_matrix=ent_matrix,
        min_similarity=leiden_min_similarity,
        min_shared_users=leiden_min_shared_users,
        entitlement_ids=matrix.columns.tolist(),
    )

    if len(edges) == 0:
        logger.warning("No edges in graph - data too sparse or thresholds too strict")
        return _empty_clustering_result(matrix)

    logger.info(f"Graph: {len(matrix.columns)} nodes, {len(edges)} edges")

    # Step 3: Run Leiden clustering
    logger.info("Step 3: Running Leiden community detection")
    entitlement_clusters, leiden_stats = _run_leiden_clustering(
        n_entitlements=len(matrix.columns),
        edges=edges,
        edge_weights=edge_weights,
        resolution=leiden_resolution,
        random_seed=leiden_random_seed,
        entitlement_ids=matrix.columns.tolist(),
    )

    logger.info(
        f"Leiden found {len(entitlement_clusters)} clusters "
        f"(modularity={leiden_stats['modularity']:.3f})"
    )

    # Step 4: Assign users to clusters (multi-membership)
    logger.info("Step 4: Assigning users to clusters (multi-membership)")
    user_memberships, cluster_user_counts = _assign_users_to_clusters(
        matrix=matrix,
        entitlement_clusters=entitlement_clusters,
        min_coverage=min_entitlement_coverage,
        max_clusters_per_user=max_clusters_per_user,
    )

    # Step 5: Drop small clusters
    logger.info("Step 5: Filtering small clusters")
    entitlement_clusters, user_memberships = _filter_small_clusters(
        entitlement_clusters=entitlement_clusters,
        user_memberships=user_memberships,
        cluster_user_counts=cluster_user_counts,
        min_role_size=min_role_size,
    )

    # Step 6: Compute metadata
    logger.info("Step 6: Computing cluster metadata")
    cluster_metadata = _compute_cluster_metadata(
        entitlement_clusters=entitlement_clusters,
        user_memberships=user_memberships,
        matrix=matrix,
    )

    # Find unassigned users
    all_users = set(matrix.index)
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
        matrix: pd.DataFrame,
        previous_result: Dict[str, Any],
        config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Incremental clustering update (for daily reclustering).

    Only re-clusters if data change exceeds threshold, otherwise updates existing.

    Args:
        matrix: Current user × entitlement matrix
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
        leiden_min_similarity=config.get("leiden_min_similarity", 0.3),
        leiden_min_shared_users=config.get("leiden_min_shared_users", 3),
        leiden_resolution=config.get("leiden_resolution", 1.0),
        leiden_random_seed=config.get("leiden_random_seed", 42),
        min_entitlement_coverage=config.get("min_entitlement_coverage", 0.5),
        max_clusters_per_user=config.get("max_clusters_per_user", 5),
        min_role_size=config.get("min_role_size", 10),
        use_sparse=config.get("use_sparse_matrices", True),
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


def _transpose_to_sparse(matrix: pd.DataFrame) -> csr_matrix:
    """Transpose matrix to sparse CSR format (entitlements × users)."""
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

    # Build edges
    edges = []
    weights = []

    for i, j, intersection in zip(overlap_coo.row, overlap_coo.col, overlap_coo.data):
        # Skip diagonal and upper triangle (undirected graph)
        if i >= j:
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
        leidenalg.ModularityVertexPartition,
        weights='weight',
        resolution_parameter=resolution,
        seed=random_seed,
        n_iterations=-1,  # Run until convergence
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
        "modularity": float(partition.modularity),
        "resolution": resolution,
        "initial_clusters": len(partition),
        "graph_nodes": n_entitlements,
        "graph_edges": len(edges),
        "avg_degree": 2 * len(edges) / n_entitlements if n_entitlements > 0 else 0,
        "isolated_nodes": n_entitlements - len(set(node for edge in edges for node in edge)),
    }

    return entitlement_clusters, leiden_stats


def _assign_users_to_clusters(
        matrix: pd.DataFrame,
        entitlement_clusters: Dict[int, List[str]],
        min_coverage: float,
        max_clusters_per_user: int,
) -> Tuple[Dict[str, List[Dict]], Dict[int, int]]:
    """
    Assign users to clusters based on coverage threshold.

    Users can belong to multiple clusters (multi-membership).

    Returns:
        user_memberships: {user_id: [{"cluster_id": X, "coverage": Y, ...}]}
        cluster_user_counts: {cluster_id: user_count}
    """
    user_memberships = {}
    cluster_user_counts = {cid: 0 for cid in entitlement_clusters.keys()}

    for user_id in matrix.index:
        # Get entitlements this user has
        user_ents = set(matrix.columns[matrix.loc[user_id] == 1])

        if not user_ents:
            continue

        memberships = []

        for cluster_id, cluster_ents in entitlement_clusters.items():
            cluster_set = set(cluster_ents)

            # Compute coverage
            intersection = user_ents & cluster_set
            coverage = len(intersection) / len(cluster_set)

            if coverage >= min_coverage:
                memberships.append({
                    "cluster_id": cluster_id,
                    "coverage": round(coverage, 4),
                    "count": len(intersection),
                    "total": len(cluster_set),
                })
                cluster_user_counts[cluster_id] += 1

        # Sort by coverage (highest first), take top N
        memberships.sort(key=lambda x: -x["coverage"])
        user_memberships[user_id] = memberships[:max_clusters_per_user]

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
        matrix: pd.DataFrame,
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


def _empty_clustering_result(matrix: pd.DataFrame) -> Dict[str, Any]:
    """Return empty result when clustering fails."""
    return {
        "entitlement_clusters": {},
        "user_cluster_membership": {},
        "cluster_metadata": {},
        "leiden_stats": {
            "modularity": 0.0,
            "initial_clusters": 0,
            "graph_nodes": len(matrix.columns),
            "graph_edges": 0,
        },
        "n_clusters": 0,
        "unassigned_users": matrix.index.tolist(),
    }


def _compute_data_change_magnitude(
        current_matrix: pd.DataFrame,
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
        matrix: pd.DataFrame,
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

    matrix = pd.DataFrame(0, index=user_ids, columns=ent_ids, dtype=np.int8)
    for u_idx, e_idx in zip(users_idx, ents_idx):
        matrix.iloc[u_idx, e_idx] = 1

    print(f"Synthetic matrix: {matrix.shape[0]} users × {matrix.shape[1]} entitlements")
    print(f"Sparsity: {1 - matrix.values.mean():.2%}")
    print(f"Avg entitlements/user: {matrix.sum(axis=1).mean():.1f}")

    # Run clustering
    if LEIDEN_AVAILABLE:
        result = cluster_entitlements_leiden(
            matrix=matrix,
            leiden_min_similarity=0.3,
            leiden_min_shared_users=3,
            leiden_resolution=1.0,
            min_entitlement_coverage=0.5,
            max_clusters_per_user=5,
            min_role_size=10,
        )

        print(f"\nClustering results:")
        print(f"  - Clusters: {result['n_clusters']}")
        print(f"  - Modularity: {result['leiden_stats']['modularity']:.3f}")
        print(f"  - Assigned users: {len(result['user_cluster_membership'])}/{n_users}")
        print(f"  - Unassigned users: {len(result['unassigned_users'])}")

        # Show cluster sizes
        print(f"\nCluster sizes:")
        for cid, meta in sorted(result['cluster_metadata'].items()):
            print(
                f"  Cluster {cid}: {meta['user_count']} users, {meta['entitlement_count']} ents, avg_coverage={meta['avg_coverage']:.2%}")
    else:
        print("\nLeiden not available - install with: pip install python-igraph leidenalg")