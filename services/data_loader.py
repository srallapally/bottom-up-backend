import os

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix


def parse_identities(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    df["USR_ID"] = df["USR_ID"].astype(str).str.strip()
    df = df.set_index("USR_ID")
    return df


def parse_entitlements(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    df["APP_ID"] = df["APP_ID"].astype(str).str.strip()
    df["ENT_ID"] = df["ENT_ID"].astype(str).str.strip()
    df["namespaced_id"] = df["APP_ID"] + ":" + df["ENT_ID"]
    return df


def parse_assignments(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    df["APP_ID"] = df["APP_ID"].astype(str).str.strip()
    df["ENT_ID"] = df["ENT_ID"].astype(str).str.strip()
    df["USR_ID"] = df["USR_ID"].astype(str).str.strip()
    df["namespaced_id"] = df["APP_ID"] + ":" + df["ENT_ID"]
    return df


def build_user_entitlement_matrix(assignments: pd.DataFrame) -> pd.DataFrame:
    """Build binary user x entitlement matrix using categorical indexing for scale."""
    user_cat = pd.Categorical(assignments["USR_ID"])
    ent_cat = pd.Categorical(assignments["namespaced_id"])

    row_idx = user_cat.codes
    col_idx = ent_cat.codes
    data = np.ones(len(assignments), dtype=np.int8)

    sparse = csr_matrix(
        (data, (row_idx, col_idx)),
        shape=(len(user_cat.categories), len(ent_cat.categories)),
    )
    # Clamp duplicates to 1
    sparse.data[:] = 1

    matrix = pd.DataFrame(
        sparse.toarray(),
        index=user_cat.categories,
        columns=ent_cat.categories,
        dtype=np.int8,
    )
    matrix.index.name = "USR_ID"
    return matrix


def process_upload(session_path: str) -> dict:
    """
    Reads uploaded CSVs from uploads/, processes them,
    saves to processed/, returns summary stats.
    """
    uploads_dir = os.path.join(session_path, "uploads")
    processed_dir = os.path.join(session_path, "processed")

    identity_file = os.path.join(uploads_dir, "identities.csv")
    assignments_file = os.path.join(uploads_dir, "assignments.csv")
    entitlements_file = os.path.join(uploads_dir, "entitlements.csv")

    # Check required files
    missing = []
    if not os.path.isfile(identity_file):
        missing.append("identities")
    if not os.path.isfile(assignments_file):
        missing.append("assignments")
    if missing:
        raise ValueError(f"Missing required files: {', '.join(missing)}")

    # Parse
    identities = parse_identities(identity_file)
    assignments = parse_assignments(assignments_file)

    catalog = None
    if os.path.isfile(entitlements_file):
        catalog = parse_entitlements(entitlements_file)

    # Build matrix
    matrix = build_user_entitlement_matrix(assignments)

    # Save processed data
    # identities.to_csv(os.path.join(processed_dir, "identities.csv"))
    identities.reset_index().to_csv(os.path.join(processed_dir, "identities.csv"), index=False)
    assignments.to_csv(os.path.join(processed_dir, "assignments.csv"), index=False)
    matrix.to_csv(os.path.join(processed_dir, "matrix.csv"))
    if catalog is not None:
        catalog.to_csv(os.path.join(processed_dir, "catalog.csv"), index=False)

    # Stats
    apps = sorted(assignments["APP_ID"].unique().tolist())
    entitlements_per_app = assignments.groupby("APP_ID")["namespaced_id"].nunique().to_dict()
    # Unique user-entitlement grants (avoid counting duplicate rows)
    total_assignments = assignments.drop_duplicates(subset=["USR_ID", "namespaced_id"]).shape[0]

    return {
        "total_users": matrix.shape[0],
        "total_entitlements": matrix.shape[1],
        "total_assignments": int(total_assignments),
        "apps": apps,
        "entitlements_per_app": entitlements_per_app,
    }