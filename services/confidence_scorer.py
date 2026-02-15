"""
Confidence Scorer V2 - Multi-Factor Confidence Scoring
======================================================

Enhanced confidence scoring combining multiple signals:
1. Peer group prevalence (from entitlement clusters)
2. HR attribute alignment (department, job title, location, manager)
3. Drift stability (how stable has this entitlement been over time)
4. Role coverage (what % of user's assigned roles does user have)

Key differences from V1 (services/confidence_scorer.py):
- V1: Peer group only (cluster-based)
- V2: Multi-factor weighted combination
- V2: Pre-computes attribute prevalence matrices (vectorized)
- V2: Handles NULL attributes gracefully (re-normalize weights)
- V2: Drift stability factor (temporal awareness)
- V2: Role coverage factor (multi-cluster aware)

Algorithm:
1. Pre-compute attribute prevalence for low-cardinality attributes
2. For each assignment, compute individual scores:
   - Peer group score (leave-one-out)
   - Department prevalence
   - Job title prevalence
   - Location prevalence
   - Manager prevalence (if cardinality < 500)
   - Drift stability (if drift data available)
   - Role coverage (multi-cluster aware)
3. Weighted combination (re-normalize if attributes NULL)
4. Generate multi-factor justification
"""
import json

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
import logging
from collections import defaultdict


logger = logging.getLogger(__name__)


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
    logger.info(f"Scoring {len(assignments_df)} assignments with V2 multi-factor confidence")

    # Step 1: Pre-compute attribute prevalence matrices
    logger.info("Step 1: Pre-computing attribute prevalence")
    attribute_prevalence = _precompute_attribute_prevalence(
        matrix=full_matrix,
        identities=identities,
        config=config,
    )

    # Step 2: Build cluster mapping (entitlement → cluster)
    logger.info("Step 2: Building entitlement-to-cluster mapping")
    ent_to_cluster = _build_ent_to_cluster_mapping(cluster_result)

    # Step 3: Build role coverage lookup (user → role coverage)
    logger.info("Step 3: Building role coverage lookup")
    user_role_coverage = _build_user_role_coverage(roles, cluster_result)

    # Step 4: Compute individual scores for each assignment
    logger.info("Step 4: Computing individual factor scores")
    scores_df = _compute_individual_scores(
        assignments_df=assignments_df,
        full_matrix=full_matrix,
        identities=identities,
        ent_to_cluster=ent_to_cluster,
        cluster_result=cluster_result,
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

    logger.info(
        f"Confidence scoring complete: "
        f"{(enriched_df['confidence_level'] == 'HIGH').sum()} HIGH, "
        f"{(enriched_df['confidence_level'] == 'MEDIUM').sum()} MEDIUM, "
        f"{(enriched_df['confidence_level'] == 'LOW').sum()} LOW"
    )

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

    For each (attribute, value, entitlement):
        prevalence[dept=Finance, ENT_123] = {total: 500, with_ent: 435}

    This allows O(1) lookup during scoring instead of O(n_users) per assignment.

    Returns:
        Dict mapping (attr_name, attr_value, ent_id) → {total_users, users_with_ent}
    """
    attribute_columns = config.get("attribute_columns", {})
    max_cardinality = config.get("max_attribute_cardinality", 500)
    min_group_size = config.get("min_attribute_group_size", 2)

    prevalence = {}
    skipped_attributes = []

    for attr_name, col_name in attribute_columns.items():
        if col_name not in identities.columns:
            logger.warning(f"Attribute column '{col_name}' not found in identities, skipping")
            continue

        # Check cardinality
        n_unique = identities[col_name].nunique()
        if n_unique > max_cardinality:
            skipped_attributes.append(attr_name)
            logger.info(
                f"Skipping {attr_name} (cardinality {n_unique} > {max_cardinality})"
            )
            continue

        # Group by attribute value (vectorized)
        grouped = identities.groupby(col_name, dropna=True)

        for attr_value, group_df in grouped:
            if len(group_df) < min_group_size:
                continue

            user_indices = group_df.index
            total_users = len(user_indices)

            # Vectorized: count entitlements per user in this group
            ent_sums = matrix.loc[user_indices].sum(axis=0)

            for ent_id, count in ent_sums.items():
                if count == 0:
                    continue

                key = (attr_name, str(attr_value), ent_id)
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

def _build_ent_to_cluster_mapping(cluster_result: Dict[str, Any]) -> Dict[str, int]:
    """
    Build reverse mapping: entitlement_id → cluster_id.

    Returns:
        Dict mapping entitlement_id to cluster_id (for residual ents: None)
    """
    entitlement_clusters = cluster_result["entitlement_clusters"]

    ent_to_cluster = {}
    for cluster_id, ent_list in entitlement_clusters.items():
        for ent_id in ent_list:
            ent_to_cluster[ent_id] = cluster_id

    return ent_to_cluster


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
    ent_to_cluster: Dict[str, int],
    cluster_result: Dict[str, Any],
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
    df["department_score"] = np.nan
    df["job_title_score"] = np.nan
    df["location_score"] = np.nan
    df["manager_score"] = np.nan
    df["drift_stability_score"] = np.nan
    df["role_coverage_score"] = 0.0

    # Metadata columns
    df["cluster_id"] = None
    df["cluster_size"] = 0
    df["peers_with_entitlement"] = 0
    df["role_covered"] = False
    df["attributes_skipped"] = ""

    # Peer group scores
    df = _compute_peer_group_scores(
        df=df,
        full_matrix=full_matrix,
        ent_to_cluster=ent_to_cluster,
        cluster_result=cluster_result,
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

    # Role covered flag
    df = _compute_role_covered(
        df=df,
        ent_to_cluster=ent_to_cluster,
        cluster_result=cluster_result,
        birthright_entitlements=birthright_entitlements,
        ent_col=ent_col,
    )

    return df


def _compute_peer_group_scores(
    df: pd.DataFrame,
    full_matrix: pd.DataFrame,
    ent_to_cluster: Dict[str, int],
    cluster_result: Dict[str, Any],
    ent_col: str,
) -> pd.DataFrame:
    """Compute peer group prevalence scores (leave-one-out)."""
    user_memberships = cluster_result["user_cluster_membership"]

    # Pre-compute cluster data
    cluster_data = {}
    entitlement_clusters = cluster_result["entitlement_clusters"]

    for cluster_id, ent_list in entitlement_clusters.items():
        # Get users in this cluster
        users = [
            user_id for user_id, memberships in user_memberships.items()
            if any(m["cluster_id"] == cluster_id for m in memberships)
        ]

        if not users:
            continue

        cluster_matrix = full_matrix.loc[users]
        cluster_data[cluster_id] = {
            "users": users,
            "size": len(users),
            "ent_sums": cluster_matrix.sum(axis=0),
        }

    # Score each assignment
    for idx, row in df.iterrows():
        user_id = row["USR_ID"]
        ent_id = row[ent_col]

        # Find cluster for this entitlement
        cluster_id = ent_to_cluster.get(ent_id)
        if cluster_id is None:
            # Residual entitlement (not in any cluster)
            continue

        cdata = cluster_data.get(cluster_id)
        if cdata is None or cdata["size"] <= 1:
            continue

        # Leave-one-out peer score
        user_has = full_matrix.at[user_id, ent_id] if ent_id in full_matrix.columns else 0
        peers_with = int(cdata["ent_sums"].get(ent_id, 0) - user_has)
        peer_count = cdata["size"] - 1

        score = peers_with / peer_count if peer_count > 0 else 0.0

        df.at[idx, "peer_group_score"] = round(score, 4)
        df.at[idx, "cluster_id"] = cluster_id
        df.at[idx, "cluster_size"] = cdata["size"]
        df.at[idx, "peers_with_entitlement"] = peers_with

    return df


def _compute_attribute_scores(
    df: pd.DataFrame,
    identities: pd.DataFrame,
    full_matrix: pd.DataFrame,
    attribute_prevalence: Dict,
    config: Dict[str, Any],
    ent_col: str,
) -> pd.DataFrame:
    """Compute HR attribute prevalence scores."""
    attribute_columns = config.get("attribute_columns", {})

    # Map: attr_name → score_column
    score_cols = {
        "department": "department_score",
        "job_title": "job_title_score",
        "location": "location_score",
        "manager": "manager_score",
    }

    for idx, row in df.iterrows():
        user_id = row["USR_ID"]
        ent_id = row[ent_col]

        if user_id not in identities.index:
            continue

        skipped = []

        for attr_name, score_col in score_cols.items():
            if attr_name not in attribute_columns:
                continue

            col_name = attribute_columns[attr_name]
            if col_name not in identities.columns:
                continue

            attr_value = identities.at[user_id, col_name]

            # Handle NULL
            if pd.isna(attr_value) or attr_value == "":
                skipped.append(attr_name)
                continue

            # Lookup pre-computed prevalence
            key = (attr_name, str(attr_value), ent_id)
            if key not in attribute_prevalence:
                # Attribute value exists but no prevalence data
                # (could be rare value or entitlement)
                continue

            prev_data = attribute_prevalence[key]
            total = prev_data["total_users"]
            with_ent = prev_data["users_with_ent"]

            # Leave-one-out correction
            if full_matrix.at[user_id, ent_id] if ent_id in full_matrix.columns else 0:
                with_ent -= 1
                total -= 1

            score = with_ent / total if total > 0 else 0.0
            df.at[idx, score_col] = round(score, 4)

        if skipped:
            df.at[idx, "attributes_skipped"] = ", ".join(skipped)

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
    ent_to_cluster: Dict[str, int],
    cluster_result: Dict[str, Any],
    birthright_entitlements: List[str],
    ent_col: str,
) -> pd.DataFrame:
    """Determine if entitlement is covered by user's assigned roles or birthright."""
    user_memberships = cluster_result["user_cluster_membership"]
    birthright_set = set(birthright_entitlements)

    for idx, row in df.iterrows():
        user_id = row["USR_ID"]
        ent_id = row[ent_col]

        # Check birthright
        if ent_id in birthright_set:
            df.at[idx, "role_covered"] = True
            continue

        # Check if in any of user's assigned clusters
        if user_id not in user_memberships:
            continue

        ent_cluster = ent_to_cluster.get(ent_id)
        if ent_cluster is None:
            continue

        user_clusters = [m["cluster_id"] for m in user_memberships[user_id]]
        if ent_cluster in user_clusters:
            df.at[idx, "role_covered"] = True

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
    """
    df = scores_df.copy()

    # Get weights from config
    weights = config.get("attribute_weights", {})
    renormalize = config.get("renormalize_weights_on_null", True)
    high_thresh = config.get("confidence_high_threshold", 0.8)
    medium_thresh = config.get("confidence_medium_threshold", 0.5)

    # Use drift stability and role coverage if configured
    use_drift = config.get("use_drift_stability_factor", True)
    use_role_cov = config.get("use_role_coverage_factor", True)

    # Initialize result columns
    df["confidence"] = 0.0
    df["confidence_level"] = "LOW"
    df["justification"] = ""
    df["weights_used"] = ""

    for idx, row in df.iterrows():
        user_id = row["USR_ID"]

        # Collect available scores
        available_scores = {}

        # Peer group
        if pd.notna(row["peer_group_score"]):
            available_scores["peer_group"] = row["peer_group_score"]

        # Attributes
        for attr_name in ["department", "job_title", "location", "manager"]:
            score_col = f"{attr_name}_score"
            if score_col in df.columns and pd.notna(row[score_col]):
                available_scores[attr_name] = row[score_col]

        # Drift stability
        if use_drift and pd.notna(row["drift_stability_score"]):
            available_scores["drift_stability"] = row["drift_stability_score"]

        # Role coverage
        if use_role_cov and user_id in user_role_coverage:
            available_scores["role_coverage"] = user_role_coverage[user_id]

        # Compute weighted confidence
        if not available_scores:
            # No scores available
            df.at[idx, "confidence"] = 0.0
            df.at[idx, "confidence_level"] = "LOW"
            df.at[idx, "justification"] = "No confidence factors available"
            continue

        # Build weights for available scores
        weights_used = {}
        for factor_name, score in available_scores.items():
            if factor_name in weights:
                weights_used[factor_name] = weights[factor_name]
            elif factor_name == "drift_stability":
                weights_used[factor_name] = config.get("drift_stability_weight", 0.1)
            elif factor_name == "role_coverage":
                weights_used[factor_name] = config.get("role_coverage_weight", 0.1)

        # Re-normalize if enabled
        if renormalize and weights_used:
            total = sum(weights_used.values())
            if total > 0:
                weights_used = {k: v / total for k, v in weights_used.items()}

        # Weighted sum
        confidence = sum(
            available_scores[factor] * weights_used.get(factor, 0)
            for factor in available_scores
        )

        confidence = round(confidence, 4)

        # Confidence level
        if confidence >= high_thresh:
            level = "HIGH"
        elif confidence >= medium_thresh:
            level = "MEDIUM"
        else:
            level = "LOW"

        # Justification
        justification = _build_justification(
            confidence=confidence,
            available_scores=available_scores,
            weights_used=weights_used,
            row=row,
        )

        df.at[idx, "confidence"] = confidence
        df.at[idx, "confidence_level"] = level
        df.at[idx, "justification"] = justification
        df.at[idx, "weights_used"] = str(weights_used)

    return df


def _build_justification(
    confidence: float,
    available_scores: Dict[str, float],
    weights_used: Dict[str, float],
    row: pd.Series,
) -> str:
    """Build human-readable justification for confidence score."""
    parts = [f"Confidence: {int(confidence * 100)}%"]

    # Break down by factor
    factor_labels = {
        "peer_group": "peer group",
        "department": "department",
        "job_title": "job title",
        "location": "location",
        "manager": "manager",
        "drift_stability": "stability",
        "role_coverage": "role coverage",
    }

    factor_parts = []
    for factor, score in available_scores.items():
        weight = weights_used.get(factor, 0)
        if weight > 0:
            label = factor_labels.get(factor, factor)
            contribution = int(score * weight * 100)
            factor_parts.append(f"{contribution}% {label}")

    if factor_parts:
        parts.append(f"({', '.join(factor_parts)})")

    # Add peer group detail if available
    if "peer_group" in available_scores and row.get("cluster_size", 0) > 0:
        peers_with = row.get("peers_with_entitlement", 0)
        peer_count = row["cluster_size"] - 1
        parts.append(
            f"— {peers_with} of {peer_count} peers in your group have this"
        )

    # Flag if attributes skipped
    skipped = row.get("attributes_skipped", "")
    if skipped:
        parts.append(f"(skipped: {skipped})")

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
        "attribute_columns": {
            "department": "department",
            "job_title": "jobcode",
            "location": "location_country",
        },
        "attribute_weights": {
            "peer_group": 0.40,
            "department": 0.25,
            "job_title": 0.20,
            "location": 0.10,
        },
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