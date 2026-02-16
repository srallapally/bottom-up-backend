"""
Role Builder V2 - Multi-Cluster Membership Role Construction
=============================================================

Builds role definitions from Leiden entitlement clusters with multi-membership support.

Key differences from V1 (services/role_builder.py):
- V1: Single-cluster membership (user belongs to 1 role)
- V2: Multi-cluster membership (user can belong to 1-5 roles)
- V1: Roles built from user clusters
- V2: Roles built from entitlement clusters
- V1: Entitlement inclusion threshold (filter entitlements)
- V2: All cluster entitlements included (coverage-based assignment already done)
- V2: Per-user coverage tracking (what % of role user has)

Algorithm:
1. For each entitlement cluster → create one role
2. Members = users assigned to this cluster (from clustering step)
3. Entitlements = all entitlements in cluster (no filtering)
4. Track per-user coverage (already computed during clustering)
5. Auto-generate role names from HR attribute dominance
6. Compute residuals (reduced vs V1 due to multi-membership)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# MAIN ROLE BUILDING FUNCTION
# ============================================================================

def build_roles(
        matrix: pd.DataFrame,
        cluster_result: Dict[str, Any],
        birthright_result: Dict[str, Any],
        identities: pd.DataFrame,
        catalog: Optional[pd.DataFrame] = None,
        config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build role definitions from entitlement clusters (V2).

    Args:
        matrix: Full user × entitlement matrix (before birthright filtering)
        cluster_result: Output from cluster_entitlements_leiden()
        birthright_result: Output from detect_birthright()
        identities: User HR attributes DataFrame
        catalog: Optional entitlement catalog (for enrichment)
        config: Optional config dict (for role naming)

    Returns:
        dict with:
            - roles: List of role objects (multi-cluster aware)
            - birthright_role: Baseline access role
            - summary: Overall statistics
            - residuals: Per-user entitlements not covered
            - multi_cluster_info: Statistics about multi-membership
    """
    config = config or {}

    logger.info("Building V2 roles from entitlement clusters")

    entitlement_clusters = cluster_result["entitlement_clusters"]
    user_memberships = cluster_result["user_cluster_membership"]
    cluster_metadata = cluster_result["cluster_metadata"]

    # Step 1: Build role per cluster (with filtering)
    min_entitlements = config.get("min_entitlements_per_role", 2)
    logger.info(f"Step 1: Building roles from {len(entitlement_clusters)} clusters "
                f"(min_entitlements={min_entitlements})")
    roles = []
    filtered_clusters = {"low_entitlements": 0, "zero_members": 0, "details": []}

    for cluster_id in sorted(entitlement_clusters.keys()):
        cluster_ents = entitlement_clusters[cluster_id]

        # Filter: too few entitlements
        if len(cluster_ents) < min_entitlements:
            filtered_clusters["low_entitlements"] += 1
            filtered_clusters["details"].append({
                "cluster_id": cluster_id,
                "reason": "low_entitlements",
                "entitlement_count": len(cluster_ents),
                "entitlements": cluster_ents,
            })
            logger.debug(
                f"Skipping cluster {cluster_id}: "
                f"{len(cluster_ents)} entitlement(s) < min {min_entitlements}"
            )
            continue

        # Filter: zero members (no users qualified for this cluster)
        has_members = any(
            any(m["cluster_id"] == cluster_id for m in memberships)
            for memberships in user_memberships.values()
        )
        if not has_members:
            filtered_clusters["zero_members"] += 1
            filtered_clusters["details"].append({
                "cluster_id": cluster_id,
                "reason": "zero_members",
                "entitlement_count": len(cluster_ents),
            })
            logger.debug(f"Skipping cluster {cluster_id}: 0 members")
            continue

        role = _build_role_from_cluster(
            cluster_id=cluster_id,
            cluster_entitlements=entitlement_clusters[cluster_id],
            user_memberships=user_memberships,
            cluster_metadata=cluster_metadata,
            matrix=matrix,
            identities=identities,
            catalog=catalog,
            config=config,
        )
        roles.append(role)

    # Step 2: Build birthright role
    logger.info("Step 2: Building birthright role")
    birthright_role = _build_birthright_role(
        birthright_result=birthright_result,
        catalog=catalog,
    )

    # Step 3: Compute residuals (per-user uncovered entitlements)
    logger.info("Step 3: Computing residual access")
    residuals = _compute_residuals_v2(
        matrix=matrix,
        roles=roles,
        user_memberships=user_memberships,
        birthright_entitlements=birthright_result["birthright_entitlements"],
    )

    # Step 4: Multi-cluster statistics
    logger.info("Step 4: Computing multi-cluster statistics")
    multi_cluster_info = _compute_multi_cluster_stats(
        user_memberships=user_memberships,
        entitlement_clusters=entitlement_clusters,
    )

    # Step 5: Naming statistics
    naming_summary = _compute_naming_summary(roles)

    # Step 6: Overall summary
    summary = _build_summary(
        roles=roles,
        user_memberships=user_memberships,
        birthright_result=birthright_result,
        residuals=residuals,
        matrix=matrix,
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
        "residuals": residuals,
        "summary": summary,
        "multi_cluster_info": multi_cluster_info,
        "naming_summary": naming_summary,
        "noise_entitlements": birthright_result.get("noise_entitlements", []),
        "filtered_clusters": filtered_clusters,
    }


# ============================================================================
# ROLE CONSTRUCTION
# ============================================================================

def _build_role_from_cluster(
        cluster_id: int,
        cluster_entitlements: List[str],
        user_memberships: Dict[str, List[Dict]],
        cluster_metadata: Dict[int, Dict],
        matrix: pd.DataFrame,
        identities: pd.DataFrame,
        catalog: Optional[pd.DataFrame],
        config: Dict[str, Any],
) -> Dict[str, Any]:
    """Build a single role from an entitlement cluster."""

    # Find users assigned to this cluster
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
                    "missing": [],  # Will populate below
                }
                break

    # Calculate what each user is missing
    cluster_set = set(cluster_entitlements)
    for user_id in members:
        user_ents = set(matrix.columns[matrix.loc[user_id] == 1])
        missing = list(cluster_set - user_ents)
        member_coverage_data[user_id]["missing"] = missing

    # Auto-generate role name
    role_name = _generate_role_name(
        members=members,
        identities=identities,
        cluster_id=cluster_id,
        config=config,
    )

    # HR attribute summary
    hr_summary = _summarize_hr_attributes(members, identities)

    # Naming source (for transparency)
    naming_source = _get_naming_source(
        members=members,
        identities=identities,
        role_name=role_name,
        cluster_id=cluster_id,
        config=config,
    )

    # Entitlement details (with catalog enrichment)
    entitlements_detail = _enrich_entitlements(
        entitlement_ids=cluster_entitlements,
        catalog=catalog,
    )

    # Coverage distribution
    coverage_dist = _compute_coverage_distribution(member_coverage_data)

    # Get cluster metadata
    meta = cluster_metadata.get(cluster_id, {})

    return {
        "role_id": f"ROLE_{cluster_id:03d}",
        "role_name": role_name,
        "entitlement_cluster_id": cluster_id,

        # Entitlements (all cluster entitlements, no filtering)
        "entitlements": cluster_entitlements,
        "entitlement_count": len(cluster_entitlements),
        "entitlements_detail": entitlements_detail,

        # Members (multi-cluster aware)
        "members": members,
        "member_count": len(members),
        "member_coverage": member_coverage_data,

        # Coverage statistics
        "avg_coverage": meta.get("avg_coverage", 0.0),
        "min_coverage": meta.get("min_coverage", 0.0),
        "max_coverage": meta.get("max_coverage", 0.0),
        "coverage_distribution": coverage_dist,

        # HR patterns
        "hr_summary": hr_summary,

        # Naming metadata
        "naming_source": naming_source,
    }


# ============================================================================
# ROLE NAMING
# ============================================================================

def _generate_role_name(
        members: List[str],
        identities: pd.DataFrame,
        cluster_id: int,
        config: Dict[str, Any],
) -> str:
    """
    Auto-generate role name from HR attribute dominance.

    Naming logic:
    1. If primary attribute ≥60% dominant AND secondary ≥60%:
       → "Engineering - DevOps Engineer"
    2. If primary attribute ≥60% dominant only:
       → "Engineering"
    3. Otherwise (no dominance):
       → "ROLE_007" (numeric fallback)

    Returns:
        Role name (semantic or numeric)
    """
    # Check if auto-naming enabled
    if not config.get("auto_generate_role_names", True):
        return f"ROLE_{cluster_id:03d}"

    if not members:
        return f"ROLE_{cluster_id:03d}"

    # Get member HR data
    member_ids = [m for m in members if m in identities.index]
    if not member_ids:
        return f"ROLE_{cluster_id:03d}"

    member_data = identities.loc[member_ids]

    # Get config params
    primary_attr = config.get("role_name_primary_attr", "department")
    secondary_attr = config.get("role_name_secondary_attr", "jobcode")
    min_dominance = config.get("role_name_min_dominance", 0.6)

    # Check primary attribute dominance
    if primary_attr in member_data.columns:
        # Filter out nulls and empty strings
        primary_values = member_data[primary_attr].dropna()
        primary_values = primary_values[primary_values != ""]

        if not primary_values.empty:
            primary_counts = primary_values.value_counts()
            top_primary = primary_counts.index[0]
            top_primary_pct = primary_counts.iloc[0] / len(member_data)

            if top_primary_pct >= min_dominance:
                # Primary dominance met
                primary_name = str(top_primary)

                # Check secondary attribute dominance
                if secondary_attr in member_data.columns:
                    secondary_values = member_data[secondary_attr].dropna()
                    secondary_values = secondary_values[secondary_values != ""]

                    if not secondary_values.empty:
                        secondary_counts = secondary_values.value_counts()
                        top_secondary = secondary_counts.index[0]
                        top_secondary_pct = secondary_counts.iloc[0] / len(member_data)

                        if top_secondary_pct >= min_dominance:
                            # Both primary and secondary met
                            return f"{primary_name} - {top_secondary}"

                # Primary only
                return primary_name

    # No dominance - numeric fallback
    return f"ROLE_{cluster_id:03d}"


def _get_naming_source(
        members: List[str],
        identities: pd.DataFrame,
        role_name: str,
        cluster_id: int,
        config: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Get metadata about how the role name was generated.

    This is for transparency/debugging - helps understand why a role
    got a semantic name vs numeric fallback.
    """
    numeric_pattern = f"ROLE_{cluster_id:03d}"

    if role_name == numeric_pattern:
        # Numeric fallback used
        return {
            "fallback": True,
            "reason": "No attribute met dominance threshold",
        }

    # Semantic name used
    member_ids = [m for m in members if m in identities.index]
    if not member_ids:
        return {"fallback": True, "reason": "No members in identities"}

    member_data = identities.loc[member_ids]

    primary_attr = config.get("role_name_primary_attr", "department")
    secondary_attr = config.get("role_name_secondary_attr", "jobcode")

    # Determine which attributes were used
    parts = role_name.split(" - ")

    if len(parts) == 2:
        # Both primary and secondary
        return {
            "fallback": False,
            "primary_attribute": primary_attr,
            "primary_value": parts[0],
            "primary_coverage": _get_attr_coverage(member_data, primary_attr, parts[0]),
            "secondary_attribute": secondary_attr,
            "secondary_value": parts[1],
            "secondary_coverage": _get_attr_coverage(member_data, secondary_attr, parts[1]),
        }
    else:
        # Primary only
        return {
            "fallback": False,
            "primary_attribute": primary_attr,
            "primary_value": role_name,
            "primary_coverage": _get_attr_coverage(member_data, primary_attr, role_name),
        }


def _get_attr_coverage(member_data: pd.DataFrame, attr: str, value: str) -> float:
    """Get % of members with this attribute value."""
    if attr not in member_data.columns:
        return 0.0

    values = member_data[attr].dropna()
    if values.empty:
        return 0.0

    count = (values == value).sum()
    return round(count / len(member_data), 4)


# ============================================================================
# HR ATTRIBUTE SUMMARIZATION
# ============================================================================

def _summarize_hr_attributes(members: List[str], identities: pd.DataFrame) -> Dict[str, List[Dict]]:
    """
    Summarize top HR attribute values for role members.

    Returns top 3 values per attribute with counts and percentages.
    """
    # Filter identities to only members (USR_ID is a column, not index)
    member_data = identities[identities['USR_ID'].isin(members)]
    if member_data.empty:
        return {}
    hr_cols = [
        "department",
        "business_unit",
        "jobcode",
        "job_level",
        "location_country",
    ]

    summary = {}

    for col in hr_cols:
        if col not in member_data.columns:
            continue

        # Filter nulls and empty strings
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
    """
    Enrich entitlement IDs with catalog metadata.

    If catalog available: lookup app_id, ent_name
    Otherwise: parse from namespaced_id (APP:ENT format)
    """
    if catalog is not None and not catalog.empty:
        # Build lookup
        catalog_lookup = {}
        print(f"DEBUG: Catalog shape: {catalog.shape}")
        print(f"DEBUG: Catalog columns: {list(catalog.columns)}")
        print(f"DEBUG: Catalog index type: {type(catalog.index)}")
        row_count = 0
        for idx, row in catalog.iterrows():
           row_count += 1
           try:
                catalog_lookup[row["namespaced_id"]] = {
                    "app_id": str(row["APP_ID"]),
                    "ent_name": str(row.get("ENT_NAME", row["ENT_ID"])),
                }
           except KeyError as e:
                print(f"ERROR at row {idx}:")
                print(f"  KeyError: {e}")
                print(f"  Row index: {row.index.tolist()}")
                print(f"  Row values: {row.tolist()}")
                raise

        print(f"DEBUG: Successfully processed {row_count} catalog rows")
        # Enrich
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
        # Parse from namespaced_id
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
    """
    Compute coverage distribution buckets.

    Returns counts per coverage range:
    - 0.50-0.60: 23 users
    - 0.60-0.70: 45 users
    - etc.
    """
    if not member_coverage:
        return {}

    coverages = [data["coverage"] for data in member_coverage.values()]

    buckets = {
        "0.50-0.60": 0,
        "0.60-0.70": 0,
        "0.70-0.80": 0,
        "0.80-0.90": 0,
        "0.90-1.00": 0,
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
        matrix: pd.DataFrame,
        roles: List[Dict],
        user_memberships: Dict[str, List[Dict]],
        birthright_entitlements: List[str],
) -> List[Dict[str, str]]:
    """
    Compute residual access (entitlements not covered by roles or birthright).

    V2 difference: Multi-cluster membership means UNION of all assigned roles.

    Returns:
        List of {USR_ID, entitlement_id} for residual assignments
    """
    # Build mapping: cluster_id → entitlements
    cluster_to_ents = {}
    for role in roles:
        cluster_id = role["entitlement_cluster_id"]
        cluster_to_ents[cluster_id] = set(role["entitlements"])

    birthright_set = set(birthright_entitlements)

    residuals = []

    # For each user with cluster membership
    for user_id, memberships in user_memberships.items():
        # Union of all assigned cluster entitlements
        covered_by_roles = set()
        for membership in memberships:
            cluster_id = membership["cluster_id"]
            cluster_ents = cluster_to_ents.get(cluster_id, set())
            covered_by_roles.update(cluster_ents)

        # Total covered = roles + birthright
        total_covered = covered_by_roles | birthright_set

        # User's actual entitlements
        user_ents = set(matrix.columns[matrix.loc[user_id] == 1])

        # Residual = actual - covered
        residual_ents = user_ents - total_covered

        for ent_id in residual_ents:
            residuals.append({
                "USR_ID": user_id,
                "entitlement_id": ent_id,
            })

    # Also check users with NO cluster membership
    all_users = set(matrix.index)
    assigned_users = set(user_memberships.keys())
    unassigned_users = all_users - assigned_users

    for user_id in unassigned_users:
        # Only covered by birthright
        user_ents = set(matrix.columns[matrix.loc[user_id] == 1])
        residual_ents = user_ents - birthright_set

        for ent_id in residual_ents:
            residuals.append({
                "USR_ID": user_id,
                "entitlement_id": ent_id,
            })

    return residuals


# ============================================================================
# STATISTICS
# ============================================================================

def _compute_multi_cluster_stats(
        user_memberships: Dict[str, List[Dict]],
        entitlement_clusters: Dict[int, List[str]],
) -> Dict[str, Any]:
    """
    Compute statistics about multi-cluster membership.

    Returns:
        - distribution: count of users with 1, 2, 3, ... clusters
        - most_common_combinations: top cluster combinations
    """
    # Distribution
    cluster_counts = {}
    for memberships in user_memberships.values():
        count = len(memberships)
        cluster_counts[count] = cluster_counts.get(count, 0) + 1

    distribution = {
        f"{count}_cluster{'s' if count > 1 else ''}": users
        for count, users in sorted(cluster_counts.items())
    }

    # Most common combinations (for multi-cluster users)
    combinations = {}
    for memberships in user_memberships.values():
        if len(memberships) <= 1:
            continue

        # Sort cluster IDs for consistent key
        cluster_ids = tuple(sorted(m["cluster_id"] for m in memberships))
        combinations[cluster_ids] = combinations.get(cluster_ids, 0) + 1

    # Top 10 combinations
    top_combinations = sorted(
        combinations.items(),
        key=lambda x: -x[1]
    )[:10]

    most_common_combinations = [
        {
            "clusters": list(cluster_ids),
            "user_count": count,
            "description": _describe_cluster_combination(cluster_ids, entitlement_clusters),
        }
        for cluster_ids, count in top_combinations
    ]

    return {
        "distribution": distribution,
        "most_common_combinations": most_common_combinations,
    }


def _describe_cluster_combination(
        cluster_ids: tuple,
        entitlement_clusters: Dict[int, List[str]],
) -> str:
    """Generate human-readable description of cluster combination."""
    # Just show cluster IDs for now
    # Could be enhanced to show dominant apps
    return f"Clusters {', '.join(str(c) for c in cluster_ids)}"


def _compute_naming_summary(roles: List[Dict]) -> Dict[str, Any]:
    """
    Compute statistics about role naming.

    Returns:
        - semantic_names: count of roles with semantic names
        - numeric_names: count of roles with numeric fallback
        - naming_coverage: % of roles with semantic names
    """
    semantic_count = 0
    numeric_count = 0

    for role in roles:
        naming_source = role.get("naming_source", {})
        if naming_source.get("fallback", False):
            numeric_count += 1
        else:
            semantic_count += 1

    total = semantic_count + numeric_count
    coverage = semantic_count / total if total > 0 else 0.0

    return {
        "semantic_names": semantic_count,
        "numeric_names": numeric_count,
        "naming_coverage": round(coverage, 4),
    }


def _build_summary(
        roles: List[Dict],
        user_memberships: Dict[str, List[Dict]],
        birthright_result: Dict[str, Any],
        residuals: List[Dict],
        matrix: pd.DataFrame,
) -> Dict[str, Any]:
    """Build overall summary statistics."""
    total_users = len(matrix)
    assigned_users = len(user_memberships)
    unassigned_users = total_users - assigned_users

    # Multi-cluster counts
    multi_cluster_users = sum(
        1 for memberships in user_memberships.values()
        if len(memberships) > 1
    )
    single_cluster_users = assigned_users - multi_cluster_users

    # Average clusters per user
    total_memberships = sum(len(m) for m in user_memberships.values())
    avg_clusters = total_memberships / assigned_users if assigned_users > 0 else 0.0

    # Coverage
    avg_coverage_values = []
    for role in roles:
        if role.get("avg_coverage"):
            avg_coverage_values.append(role["avg_coverage"])

    overall_avg_coverage = np.mean(avg_coverage_values) if avg_coverage_values else 0.0

    return {
        "total_roles": len(roles),
        "total_users": total_users,
        "assigned_users": assigned_users,
        "unassigned_users": unassigned_users,
        "multi_cluster_users": multi_cluster_users,
        "single_cluster_users": single_cluster_users,
        "avg_clusters_per_user": round(avg_clusters, 2),
        "max_clusters_per_user": max(
            (len(m) for m in user_memberships.values()),
            default=0
        ),
        "avg_coverage": round(overall_avg_coverage, 4),
        "birthright_entitlements": len(birthright_result["birthright_entitlements"]),
        "noise_entitlements": len(birthright_result.get("noise_entitlements", [])),
        "residual_assignments": len(residuals),
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    import sys

    sys.path.insert(0, '/home/claude')

    # Example with synthetic data
    print("Role Builder V2 - Example Usage")
    print("=" * 70)

    # Simulate clustering result
    cluster_result = {
        "entitlement_clusters": {
            1: ["AWS:S3_Read", "AWS:S3_Write", "AWS:EC2_Admin"],
            2: ["Salesforce:Account_View", "Salesforce:Lead_Edit"],
        },
        "user_cluster_membership": {
            "USR_001": [
                {"cluster_id": 1, "coverage": 0.67, "count": 2, "total": 3},
            ],
            "USR_002": [
                {"cluster_id": 1, "coverage": 1.00, "count": 3, "total": 3},
                {"cluster_id": 2, "coverage": 0.50, "count": 1, "total": 2},
            ],
        },
        "cluster_metadata": {
            1: {"avg_coverage": 0.84, "min_coverage": 0.67, "max_coverage": 1.00},
            2: {"avg_coverage": 0.50, "min_coverage": 0.50, "max_coverage": 0.50},
        },
    }

    # Simulate birthright
    birthright_result = {
        "birthright_entitlements": ["AD:User"],
        "birthright_stats": {"AD:User": {"count": 1000, "pct": 0.99}},
        "noise_entitlements": [],
    }

    # Simulate matrix
    matrix = pd.DataFrame({
        "AWS:S3_Read": [1, 1],
        "AWS:S3_Write": [1, 1],
        "AWS:EC2_Admin": [0, 1],
        "Salesforce:Account_View": [0, 1],
        "Salesforce:Lead_Edit": [0, 0],
        "AD:User": [1, 1],
    }, index=["USR_001", "USR_002"])

    # Simulate identities
    identities = pd.DataFrame({
        "department": ["Engineering", "Engineering"],
        "jobcode": ["DevOps Engineer", "DevOps Engineer"],
    }, index=["USR_001", "USR_002"])

    # Build roles
    result = build_roles_v2(
        matrix=matrix,
        cluster_result=cluster_result,
        birthright_result=birthright_result,
        identities=identities,
        catalog=None,
        config={
            "auto_generate_role_names": True,
            "role_name_primary_attr": "department",
            "role_name_secondary_attr": "jobcode",
            "role_name_min_dominance": 0.6,
        },
    )

    print("\nResults:")
    print(f"  Roles: {len(result['roles'])}")
    print(f"  Assigned users: {result['summary']['assigned_users']}")
    print(f"  Multi-cluster users: {result['summary']['multi_cluster_users']}")
    print(f"  Residuals: {result['summary']['residual_assignments']}")

    print("\nRole 1:")
    role1 = result['roles'][0]
    print(f"  ID: {role1['role_id']}")
    print(f"  Name: {role1['role_name']}")
    print(f"  Members: {role1['member_count']}")
    print(f"  Avg coverage: {role1['avg_coverage']:.2%}")
    print(f"  Naming: {role1['naming_source']}")

    print("\nMulti-cluster info:")
    for k, v in result['multi_cluster_info']['distribution'].items():
        print(f"  {k}: {v} users")

    print("\n✓ Role builder V2 example complete")