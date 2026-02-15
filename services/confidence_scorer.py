"""
Confidence Scorer Service
=========================

Cluster-based peer confidence scoring for the role mining pipeline.
Integrates as Step 4 after role building.

Scoring formula (leave-one-out):
    confidence(U, E) = |peers in C with E| / |peers in C|
    where peers = members of cluster C, excluding user U
"""

import json
import os
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 4a: Score existing assignments
# ---------------------------------------------------------------------------

def score_assignments(
    full_matrix: pd.DataFrame,
    cluster_labels: pd.Series,
    assignments_df: pd.DataFrame,
    birthright_entitlements: list[str],
    noise_entitlements: list[str],
    roles: list[dict],
    config: dict,
    identities: pd.DataFrame,
) -> pd.DataFrame:
    """
    Enrich every row in assignments_df with confidence score, level,
    and human-readable justification.

    Returns a copy of assignments_df with added columns:
        confidence, confidence_level, global_prevalence, role_covered,
        cluster_id, cluster_size, peers_with_entitlement,
        low_peer_count, justification
    """
    high = config.get("confidence_high_threshold", 0.8)
    medium = config.get("confidence_medium_threshold", 0.5)
    min_user = config.get("min_user_assignments", 3)
    min_peers = config.get("min_reliable_peer_count", 5)

    df = assignments_df.copy()
    n_users = full_matrix.shape[0]

    # Global prevalence: % of all users holding each entitlement
    global_prev = full_matrix.sum(axis=0) / n_users

    # Identify which users qualify for cluster-based scoring
    # Count non-birthright assignments per user
    non_br_cols = [c for c in full_matrix.columns if c not in birthright_entitlements]
    if non_br_cols:
        non_br_counts = full_matrix[non_br_cols].sum(axis=1)
    else:
        non_br_counts = pd.Series(0, index=full_matrix.index)

    users_below_min = set(non_br_counts[non_br_counts < min_user].index)

    # Pre-compute cluster data
    assigned_mask = cluster_labels.notna()
    cluster_data = {}
    for cid in cluster_labels[assigned_mask].unique():
        members = cluster_labels[cluster_labels == cid].index.tolist()
        member_mat = full_matrix.loc[members]
        cluster_data[int(cid)] = {
            "members": members,
            "size": len(members),
            "ent_sums": member_mat.sum(axis=0),
        }

    # Role coverage lookup: for each cluster, which entitlements are in its role
    role_ent_sets = {}
    for role in roles:
        role_ent_sets[role["role_id"]] = set(role.get("entitlements", []))

    # Map cluster_id -> role_id (roles are built from clusters, ROLE_001 = cluster 1)
    cluster_to_role_ents = {}
    for role in roles:
        # Extract cluster number from role_id like "ROLE_001" -> 1
        try:
            cnum = int(role["role_id"].split("_")[1])
            cluster_to_role_ents[cnum] = role_ent_sets[role["role_id"]]
        except (IndexError, ValueError):
            pass

    # Birthright entitlement set for role_covered check
    br_set = set(birthright_entitlements)

    # Initialize output columns
    out_confidence = np.zeros(len(df), dtype=np.float64)
    out_level = [""] * len(df)
    out_global = np.zeros(len(df), dtype=np.float64)
    out_covered = [False] * len(df)
    out_cluster = [None] * len(df)
    out_csize = [None] * len(df)
    out_peers = np.zeros(len(df), dtype=np.int32)
    out_low_peer = [False] * len(df)
    out_justification = [""] * len(df)

    # Build a quick lookup: namespaced_id column name
    ent_col = "namespaced_id" if "namespaced_id" in df.columns else "ENT_ID"

    for idx, (_, row) in enumerate(df.iterrows()):
        usr = row["USR_ID"]
        ent = row[ent_col]

        # Global prevalence
        gp = float(global_prev.get(ent, 0.0))
        out_global[idx] = round(gp, 4)

        # Determine scoring path
        if ent in br_set:
            # Birthright: always HIGH
            out_confidence[idx] = round(gp, 4)
            out_level[idx] = "HIGH"
            out_covered[idx] = True
            pct = int(round(gp * 100))
            out_justification[idx] = f"Held by {pct}% of all users in the organization"

        elif ent in noise_entitlements:
            # Noise: definitive LOW
            out_confidence[idx] = round(gp, 4)
            out_level[idx] = "LOW"
            total_holders = int(round(gp * n_users))
            out_justification[idx] = (
                f"Only {total_holders} user{'s' if total_holders != 1 else ''} "
                f"across the organization hold this entitlement"
            )
            # role_covered: check if any role includes it
            label_val = cluster_labels.get(usr)
            if label_val is not None and not pd.isna(label_val):
                cid = int(label_val)
                out_covered[idx] = ent in cluster_to_role_ents.get(cid, set())

        elif usr in users_below_min:
            # User below min_user_assignments: no cluster scoring
            out_confidence[idx] = 0.0
            out_level[idx] = "LOW"
            out_justification[idx] = (
                f"User has fewer than {min_user} non-birthright entitlements "
                f"\u2014 insufficient for peer group scoring"
            )

        else:
            # Cluster-based scoring
            label_val = cluster_labels.get(usr)
            if label_val is None or pd.isna(label_val):
                out_confidence[idx] = 0.0
                out_level[idx] = "LOW"
                out_justification[idx] = "User is not assigned to any peer group"
                continue

            cid = int(label_val)
            cinfo = cluster_data.get(cid)
            if cinfo is None:
                out_confidence[idx] = 0.0
                out_level[idx] = "LOW"
                out_justification[idx] = "User is not assigned to any peer group"
                continue

            out_cluster[idx] = cid
            out_csize[idx] = cinfo["size"]
            low_peer = cinfo["size"] < min_peers
            out_low_peer[idx] = low_peer

            if cinfo["size"] <= 1:
                out_confidence[idx] = 0.0
                out_level[idx] = "LOW"
                out_justification[idx] = "No peers in this group for comparison"
            else:
                user_val = full_matrix.at[usr, ent] if ent in full_matrix.columns else 0
                peers_with = int(cinfo["ent_sums"].get(ent, 0) - user_val)
                peer_count = cinfo["size"] - 1
                conf = peers_with / peer_count

                out_confidence[idx] = round(conf, 4)
                out_peers[idx] = peers_with

                if conf >= high:
                    out_level[idx] = "HIGH"
                elif conf >= medium:
                    out_level[idx] = "MEDIUM"
                else:
                    out_level[idx] = "LOW"

                just = (
                    f"{peers_with} of {peer_count} users in your peer group "
                    f"hold this entitlement"
                )
                if low_peer:
                    just += (
                        f" (confidence may be less reliable \u2014 "
                        f"only {cinfo['size']} users in this peer group)"
                    )
                out_justification[idx] = just

            # role_covered
            out_covered[idx] = ent in cluster_to_role_ents.get(cid, set()) or ent in br_set

    df["confidence"] = out_confidence
    df["confidence_level"] = out_level
    df["global_prevalence"] = out_global
    df["role_covered"] = out_covered
    df["cluster_id"] = out_cluster
    df["cluster_size"] = out_csize
    df["peers_with_entitlement"] = out_peers
    df["low_peer_count"] = out_low_peer
    df["justification"] = out_justification

    return df


# ---------------------------------------------------------------------------
# 4b: Reconstruction fit (also called inside score_assignments for the
#     role_covered column, but exposed separately for flexibility)
# ---------------------------------------------------------------------------

def check_reconstruction_fit(
    assignments_df: pd.DataFrame,
    roles: list[dict],
    cluster_labels: pd.Series,
    birthright_entitlements: list[str],
) -> pd.Series:
    """
    Returns a boolean Series indicating whether each assignment is
    covered by the user's assigned role or by birthright.
    """
    br_set = set(birthright_entitlements)
    cluster_to_role_ents = {}
    for role in roles:
        try:
            cnum = int(role["role_id"].split("_")[1])
            cluster_to_role_ents[cnum] = set(role.get("entitlements", []))
        except (IndexError, ValueError):
            pass

    ent_col = "namespaced_id" if "namespaced_id" in assignments_df.columns else "ENT_ID"
    results = []

    for _, row in assignments_df.iterrows():
        usr = row["USR_ID"]
        ent = row[ent_col]

        if ent in br_set:
            results.append(True)
            continue

        label_val = cluster_labels.get(usr)
        if label_val is None or pd.isna(label_val):
            results.append(False)
            continue

        cid = int(label_val)
        results.append(ent in cluster_to_role_ents.get(cid, set()))

    return pd.Series(results, index=assignments_df.index)


# ---------------------------------------------------------------------------
# 4c: Generate recommendations for missing entitlements
# ---------------------------------------------------------------------------

def generate_recommendations(
    full_matrix: pd.DataFrame,
    cluster_labels: pd.Series,
    assignments_df: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """
    For each clustered user, find entitlements they don't hold but peers do,
    above recommendation_min_confidence.

    Returns DataFrame with: USR_ID, APP_ID, ENT_ID, confidence,
    confidence_level, global_prevalence, cluster_id, cluster_size,
    peers_with_entitlement, low_peer_count, sod_warning, justification
    """
    min_conf = config.get("recommendation_min_confidence", 0.6)
    high = config.get("confidence_high_threshold", 0.8)
    medium = config.get("confidence_medium_threshold", 0.5)
    min_peers = config.get("min_reliable_peer_count", 5)
    n_users = full_matrix.shape[0]

    global_prev = full_matrix.sum(axis=0) / n_users

    # Pre-compute cluster data
    assigned_mask = cluster_labels.notna()
    assigned_users = cluster_labels[assigned_mask]

    cluster_data = {}
    for cid in assigned_users.unique():
        members = assigned_users[assigned_users == cid].index.tolist()
        member_mat = full_matrix.loc[members]
        cluster_data[int(cid)] = {
            "members": members,
            "size": len(members),
            "ent_sums": member_mat.sum(axis=0),
        }

    # Build lookup of what each user currently holds
    ent_col = "namespaced_id" if "namespaced_id" in assignments_df.columns else "ENT_ID"
    user_ents = assignments_df.groupby("USR_ID")[ent_col].apply(set).to_dict()

    # Namespace mapping for splitting back to APP_ID:ENT_ID
    all_ents = full_matrix.columns.tolist()

    rows = []
    for cid, cinfo in cluster_data.items():
        if cinfo["size"] <= 1:
            continue

        low_peer = cinfo["size"] < min_peers

        for usr in cinfo["members"]:
            held = user_ents.get(usr, set())
            user_val = full_matrix.loc[usr]
            peer_count = cinfo["size"] - 1

            for ent in all_ents:
                if ent in held:
                    continue
                if user_val.get(ent, 0) == 1:
                    continue  # already has it even if not in assignments_df

                peers_with = int(cinfo["ent_sums"].get(ent, 0) - user_val.get(ent, 0))
                conf = peers_with / peer_count

                if conf < min_conf:
                    continue

                if conf >= high:
                    level = "HIGH"
                elif conf >= medium:
                    level = "MEDIUM"
                else:
                    continue  # only MEDIUM and HIGH

                gp = float(global_prev.get(ent, 0.0))

                just = (
                    f"{peers_with} of {peer_count} users in your peer group "
                    f"hold this entitlement"
                )
                if low_peer:
                    just += (
                        f" (confidence may be less reliable \u2014 "
                        f"only {cinfo['size']} users in this peer group)"
                    )

                # Split namespaced_id -> APP_ID, ENT_ID
                parts = ent.split(":", 1)
                app_id = parts[0] if len(parts) == 2 else ""
                ent_id = parts[1] if len(parts) == 2 else ent

                rows.append({
                    "USR_ID": usr,
                    "APP_ID": app_id,
                    "ENT_ID": ent_id,
                    "confidence": round(conf, 4),
                    "confidence_level": level,
                    "global_prevalence": round(gp, 4),
                    "cluster_id": cid,
                    "cluster_size": cinfo["size"],
                    "peers_with_entitlement": peers_with,
                    "low_peer_count": low_peer,
                    "sod_warning": None,
                    "justification": just,
                })

    recs = pd.DataFrame(rows)
    if not recs.empty:
        recs = recs.sort_values(
            ["USR_ID", "confidence"], ascending=[True, False]
        ).reset_index(drop=True)

    return recs


# ---------------------------------------------------------------------------
# 4d: SoD co-occurrence check on recommendations
# ---------------------------------------------------------------------------

def check_sod_conflicts(
    recommendations: pd.DataFrame,
    full_matrix: pd.DataFrame,
    assignments_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Identify entitlement pairs that rarely co-occur (implicit SoD).
    If a recommendation would create such a pair with the user's existing
    access, populate sod_warning.

    Returns recommendations DataFrame with sod_warning column updated.
    """
    if recommendations.empty:
        return recommendations

    recs = recommendations.copy()
    n_users = full_matrix.shape[0]

    # Compute pairwise co-occurrence for entitlements that appear in recommendations
    rec_ents_namespaced = set()
    for _, row in recs.iterrows():
        ns = f"{row['APP_ID']}:{row['ENT_ID']}" if row["APP_ID"] else row["ENT_ID"]
        rec_ents_namespaced.add(ns)

    # Get prevalence of each entitlement
    ent_prev = full_matrix.sum(axis=0) / n_users

    # For each recommended entitlement, check co-occurrence with user's held entitlements
    ent_col = "namespaced_id" if "namespaced_id" in assignments_df.columns else "ENT_ID"
    user_ents = assignments_df.groupby("USR_ID")[ent_col].apply(set).to_dict()

    # Pre-compute co-occurrence counts for relevant pairs
    mat = full_matrix.values.astype(np.float64)
    col_index = {col: i for i, col in enumerate(full_matrix.columns)}

    warnings = [None] * len(recs)

    for idx, row in recs.iterrows():
        ns_rec = f"{row['APP_ID']}:{row['ENT_ID']}" if row["APP_ID"] else row["ENT_ID"]
        if ns_rec not in col_index:
            continue

        rec_col = col_index[ns_rec]
        rec_prev = ent_prev.get(ns_rec, 0)
        if rec_prev == 0:
            continue

        usr = row["USR_ID"]
        held = user_ents.get(usr, set())

        for held_ent in held:
            if held_ent not in col_index:
                continue

            held_col = col_index[held_ent]
            held_prev = ent_prev.get(held_ent, 0)
            if held_prev == 0:
                continue

            # Expected co-occurrence under independence
            expected = rec_prev * held_prev * n_users

            if expected < 1:
                continue  # too rare to judge

            # Observed co-occurrence
            observed = float(np.sum(mat[:, rec_col] * mat[:, held_col]))

            # If observed is significantly below expected, flag
            ratio = observed / expected
            if ratio < 0.1 and observed < 3:
                # Split held_ent for display
                parts = held_ent.split(":", 1)
                display_ent = parts[1] if len(parts) == 2 else held_ent
                display_app = parts[0] if len(parts) == 2 else ""
                label = f"{display_app}:{display_ent}" if display_app else display_ent

                loc = recs.index.get_loc(idx)
                warnings[loc] = (
                    f"This entitlement rarely co-occurs with {label} "
                    f"which you already hold \u2014 may conflict with "
                    f"separation of duty requirements"
                )
                break  # one warning per recommendation is enough

    recs["sod_warning"] = warnings
    return recs


# ---------------------------------------------------------------------------
# 4e: Over-provisioned access detection
# ---------------------------------------------------------------------------

def detect_over_provisioned(
    enriched_assignments: pd.DataFrame,
    revocation_threshold: float,
) -> pd.DataFrame:
    """
    Filter enriched assignments for rows below revocation_threshold.
    These are access accumulation / revocation candidates.
    """
    mask = enriched_assignments["confidence"] < revocation_threshold

    # Exclude birthright (they're always HIGH and shouldn't be flagged)
    if "confidence_level" in enriched_assignments.columns:
        # Birthright justifications start with "Held by"
        br_mask = enriched_assignments["justification"].str.startswith("Held by", na=False)
        mask = mask & ~br_mask

    result = enriched_assignments[mask].copy()

    if not result.empty:
        result = result.sort_values(
            ["USR_ID", "confidence"], ascending=[True, True]
        ).reset_index(drop=True)

    return result


# ---------------------------------------------------------------------------
# 4f: Cluster diagnostics (backend only)
# ---------------------------------------------------------------------------

def compute_cluster_diagnostics(
    cluster_labels: pd.Series,
    identities: pd.DataFrame,
    previous_labels_path: Optional[str] = None,
) -> dict:
    """
    Compute cluster quality metrics (not exposed to users).

    Returns dict with:
        - clusters: per-cluster homogeneity data
        - stability: comparison with prior run (if available)
    """
    assigned_mask = cluster_labels.notna()
    assigned = cluster_labels[assigned_mask]

    diagnostics = {"clusters": {}, "stability": None}

    # --- Homogeneity ---
    # Find categorical columns in identities
    cat_cols = []
    for col in identities.columns:
        if identities[col].dtype == "object" or identities[col].dtype.name == "category":
            nunique = identities[col].nunique()
            if 2 <= nunique <= 100:  # reasonable categorical range
                cat_cols.append(col)

    for cid in sorted(assigned.unique()):
        members = assigned[assigned == cid].index.tolist()
        cinfo = {
            "cluster_id": int(cid),
            "size": len(members),
            "dominant_attributes": {},
            "homogeneity_score": None,
        }

        if cat_cols:
            concentrations = []
            member_ids_in_ident = [m for m in members if m in identities.index]

            if member_ids_in_ident:
                for col in cat_cols:
                    vals = identities.loc[member_ids_in_ident, col].dropna()
                    if vals.empty:
                        continue
                    mode_val = vals.mode().iloc[0] if not vals.mode().empty else None
                    mode_pct = (vals == mode_val).mean() if mode_val is not None else 0
                    cinfo["dominant_attributes"][col] = {
                        "value": str(mode_val),
                        "percentage": round(float(mode_pct), 4),
                    }
                    concentrations.append(float(mode_pct))

            if concentrations:
                cinfo["homogeneity_score"] = round(
                    sum(concentrations) / len(concentrations), 4
                )

        diagnostics["clusters"][str(int(cid))] = cinfo

    # --- Stability ---
    if previous_labels_path and os.path.isfile(previous_labels_path):
        try:
            with open(previous_labels_path, "r") as f:
                prev_data = json.load(f)

            prev_labels = pd.Series(prev_data)
            prev_labels.index = prev_labels.index.astype(str)

            # Compare only users present in both runs
            common = assigned.index.intersection(prev_labels.index)
            if len(common) > 10:
                curr = assigned.loc[common].values
                prev = prev_labels.loc[common].values

                # Simple overlap: % of users in same cluster
                # (cluster IDs may differ, so use Adjusted Rand Index)
                try:
                    from sklearn.metrics import adjusted_rand_score
                    ari = adjusted_rand_score(prev.astype(str), curr.astype(str))
                except ImportError:
                    ari = None

                # Also compute naive stability (same label %)
                same = (curr.astype(str) == prev.astype(str)).mean()

                diagnostics["stability"] = {
                    "stability_ari": round(float(ari), 4) if ari is not None else None,
                    "pct_users_stable": round(float(same), 4),
                    "common_users": len(common),
                    "previous_run_clusters": int(prev_labels.nunique()),
                }
        except (json.JSONDecodeError, KeyError, Exception):
            diagnostics["stability"] = None

    return diagnostics


# ---------------------------------------------------------------------------
# Save / load cluster assignments for stability tracking
# ---------------------------------------------------------------------------

def save_cluster_assignments(cluster_labels: pd.Series, path: str):
    """Save cluster assignments to JSON for future stability comparison."""
    assigned = cluster_labels[cluster_labels.notna()]
    data = {str(k): int(v) for k, v in assigned.items()}
    with open(path, "w") as f:
        json.dump(data, f)


# ---------------------------------------------------------------------------
# Scoring summary for results.json
# ---------------------------------------------------------------------------

def build_scoring_summary(
    enriched: pd.DataFrame,
    recommendations: pd.DataFrame,
    over_provisioned: pd.DataFrame,
    cluster_labels: pd.Series,
    birthright_entitlements: list[str],
) -> dict:
    """Build summary stats for the confidence scoring run."""
    assigned = cluster_labels[cluster_labels.notna()]

    # Birthright-only users: have assignments but no cluster
    all_users = enriched["USR_ID"].unique()
    clustered_users = set(assigned.index)
    birthright_only = [u for u in all_users if u not in clustered_users]

    sod_warnings = 0
    if not recommendations.empty and "sod_warning" in recommendations.columns:
        sod_warnings = int(recommendations["sod_warning"].notna().sum())

    return {
        "confidence_scoring": {
            "total_scored_assignments": len(enriched),
            "high": int((enriched["confidence_level"] == "HIGH").sum()),
            "medium": int((enriched["confidence_level"] == "MEDIUM").sum()),
            "low": int((enriched["confidence_level"] == "LOW").sum()),
            "recommendations": {
                "total": len(recommendations),
                "high": int((recommendations["confidence_level"] == "HIGH").sum()) if not recommendations.empty else 0,
                "medium": int((recommendations["confidence_level"] == "MEDIUM").sum()) if not recommendations.empty else 0,
                "unique_users": int(recommendations["USR_ID"].nunique()) if not recommendations.empty else 0,
                "sod_warnings": sod_warnings,
            },
            "over_provisioned": {
                "total": len(over_provisioned),
                "unique_users": int(over_provisioned["USR_ID"].nunique()) if not over_provisioned.empty else 0,
                "not_role_covered": int((~over_provisioned["role_covered"]).sum()) if not over_provisioned.empty and "role_covered" in over_provisioned.columns else 0,
            },
            "birthright_only_users": {
                "count": len(birthright_only),
                "user_ids": birthright_only,
            },
        }
    }