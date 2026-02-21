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
    import logging
    _logger = logging.getLogger(__name__)
    _logger.info(f"Parsing {filepath}")
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    raw_count = len(df)
    df = df.dropna(subset=["USR_ID", "APP_ID", "ENT_ID"])
    dropped = raw_count - len(df)
    if dropped > 0:
        _logger.warning(
            f"parse_assignments: dropped {dropped} rows with null USR_ID/APP_ID/ENT_ID (raw={raw_count}, clean={len(df)})")
    else:
        _logger.info(f"parse_assignments: {raw_count} rows, none dropped")

    df["APP_ID"] = df["APP_ID"].astype(str).str.strip()
    df["ENT_ID"] = df["ENT_ID"].astype(str).str.strip()
    df["USR_ID"] = df["USR_ID"].astype(str).str.strip()
    df["namespaced_id"] = df["APP_ID"] + ":" + df["ENT_ID"]
    return df


def build_user_entitlement_matrix(assignments: pd.DataFrame):
    """
    Build binary user x entitlement matrix using categorical indexing for scale.
    
    CHANGE 2026-02-17: Now returns sparse matrix + indices to avoid densification.
    
    Returns:
        tuple: (sparse_matrix, user_ids, ent_ids) where:
            - sparse_matrix: scipy.sparse.csr_matrix (users × entitlements)
            - user_ids: Index object with user IDs (row labels)
            - ent_ids: Index object with entitlement IDs (column labels)
    """
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

    # CHANGE 2026-02-17: Return sparse + indices, not dense DataFrame
    return sparse, user_cat.categories, ent_cat.categories


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
    # CHANGE 2026-02-17: Now returns sparse matrix + indices
    matrix_sparse, user_ids, ent_ids = build_user_entitlement_matrix(assignments)

    # Save processed data
    identities.reset_index().to_csv(os.path.join(processed_dir, "identities.csv"), index=False)
    assignments.to_csv(os.path.join(processed_dir, "assignments.csv"), index=False)
    
    # CHANGE 2026-02-17: Save sparse matrix in npz format instead of CSV
    # Also save indices separately for reconstruction
    import scipy.sparse as sp
    sp.save_npz(os.path.join(processed_dir, "matrix.npz"), matrix_sparse)
    pd.Series(user_ids, name="USR_ID").to_csv(
        os.path.join(processed_dir, "matrix_users.csv"), index=False
    )
    pd.Series(ent_ids, name="namespaced_id").to_csv(
        os.path.join(processed_dir, "matrix_entitlements.csv"), index=False
    )
    
    if catalog is not None:
        catalog.to_csv(os.path.join(processed_dir, "catalog.csv"), index=False)

    # Stats
    apps = sorted(assignments["APP_ID"].unique().tolist())
    entitlements_per_app = assignments.groupby("APP_ID")["namespaced_id"].nunique().to_dict()
    # Unique user-entitlement grants (avoid counting duplicate rows)
    total_assignments = assignments.drop_duplicates(subset=["USR_ID", "namespaced_id"]).shape[0]

    return {
        "total_users": matrix_sparse.shape[0],
        "total_entitlements": matrix_sparse.shape[1],
        "total_assignments": int(total_assignments),
        "apps": apps,
        "entitlements_per_app": entitlements_per_app,
    }