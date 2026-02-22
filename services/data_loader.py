# services/data_loader.py
import logging
import os

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

_logger = logging.getLogger(__name__)


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
    """
    Parse assignments CSV.

    Cleaning steps (Fix 1, Fix 2):
      1. Strip whitespace from ID columns before any filtering so
         whitespace-only values ("  ") don't survive the null check.
      2. Replace empty strings with NA after stripping, then drop nulls.
      3. Deduplicate on (USR_ID, namespaced_id) so the saved CSV is
         clean and the confidence scorer doesn't double-count grants.
    """
    _logger.info("parse_assignments: reading %s", filepath)
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    raw_count = len(df)

    # FIX 1: Strip before drop so whitespace-only IDs are caught.
    for col in ("USR_ID", "APP_ID", "ENT_ID"):
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().replace("", pd.NA)

    df = df.dropna(subset=["USR_ID", "APP_ID", "ENT_ID"])
    after_drop = len(df)
    dropped_null = raw_count - after_drop
    if dropped_null > 0:
        _logger.warning(
            "parse_assignments: dropped %d junk rows (null/empty IDs) "
            "raw=%d clean=%d",
            dropped_null, raw_count, after_drop,
        )
    else:
        _logger.info("parse_assignments: %d rows, no junk rows dropped", raw_count)

    df["namespaced_id"] = df["APP_ID"] + ":" + df["ENT_ID"]

    # FIX 2: Deduplicate before saving so downstream consumers (confidence
    # scorer) see one row per user-entitlement grant.
    before_dedup = len(df)
    df = df.drop_duplicates(subset=["USR_ID", "namespaced_id"])
    dropped_dupes = before_dedup - len(df)
    if dropped_dupes > 0:
        _logger.warning(
            "parse_assignments: dropped %d duplicate (USR_ID, namespaced_id) rows "
            "before_dedup=%d after_dedup=%d",
            dropped_dupes, before_dedup, len(df),
        )
    else:
        _logger.info("parse_assignments: no duplicate grants found")

    _logger.info(
        "parse_assignments: final=%d rows, %d unique users, %d unique entitlements",
        len(df),
        df["USR_ID"].nunique(),
        df["namespaced_id"].nunique(),
    )
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
    # Clamp any residual duplicates to 1 (assignments is already deduped,
    # but csr_matrix sums repeated (row, col) pairs during construction).
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
    # FIX 2: assignments is already deduplicated by parse_assignments; save as-is.
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

    # Stats — assignments is already deduplicated so len() is correct directly.
    apps = sorted(assignments["APP_ID"].unique().tolist())
    entitlements_per_app = assignments.groupby("APP_ID")["namespaced_id"].nunique().to_dict()
    total_assignments = len(assignments)

    _logger.info(
        "process_upload: users=%d entitlements=%d assignments=%d apps=%d",
        matrix_sparse.shape[0],
        matrix_sparse.shape[1],
        total_assignments,
        len(apps),
    )

    return {
        "total_users": matrix_sparse.shape[0],
        "total_entitlements": matrix_sparse.shape[1],
        "total_assignments": total_assignments,
        "apps": apps,
        "entitlements_per_app": entitlements_per_app,
    }