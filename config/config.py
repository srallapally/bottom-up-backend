# config/config.py
"""
Role Mining V2 Configuration
Hybrid approach with daily reclustering, business role governance, and drift detection.

Integration with session-based Flask API (no database, CSV storage).
"""

from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field, asdict
import json
import os

BASE_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sessions")

# ============================================================================
# CONFIGURATION VALIDATION RULES
# ============================================================================

VALIDATION_RULES = {
    "leiden_min_similarity": {
        "min": 0.2,
        "max": 0.7,
        "error_low": "Will create giant cluster (most entitlements similar)",
        "error_high": "Will create too many singletons (few edges)",
    },
    "leiden_min_shared_users": {
        "min": 2,
        "error": "Must be >= 2 for meaningful co-occurrence",
    },
    "leiden_resolution": {
        "min": 0.3,
        "max": 3.0,
        "warning_low": "Resolution < 0.5 may merge distinct roles",
        "warning_high": "Resolution > 2.0 may over-segment",
    },
    "min_entitlement_coverage": {
        "min": 0.3,
        "max": 0.8,
        "error_low": "Coverage < 0.3 assigns users to too many roles",
        "error_high": "Coverage > 0.8 leaves most users unassigned",
    },
    # CHANGED: removed attribute_weights rule, replaced with user_attributes
    "user_attributes": {
        "min_count": 2,
        "max_count": 5,
        "weight_sum": 1.0,
        "weight_tolerance": 0.01,
        "error_count": "Must specify 2-5 user attributes",
        "error_weight_sum": "Attribute weights must sum to 1.0",
        "error_weight_range": "Each attribute weight must be > 0 and <= 1.0",
        "error_duplicate": "Duplicate attribute column names not allowed",
        "error_missing_column": "Each attribute must have a 'column' field",
    },
    "max_clusters_per_user": {
        "min": 1,
        "max": 10,
        "warning_high": "Max > 5 may indicate noisy assignments",
    },
    "min_entitlements_per_role": {
        "min": 2,
        "error": "Must be >= 2 (single-entitlement roles are not meaningful)",
    },
    # ADDED: prevalence-based role overlap parameters
    "entitlement_prevalence_threshold": {
        "min": 0.5,
        "max": 1.0,
        "error_low": "Prevalence threshold < 0.5 produces noisy core sets",
        "error_high": "Prevalence threshold cannot exceed 1.0",
    },
    "entitlement_association_threshold": {
        "min": 0.2,
        "max": 0.8,
        "error_low": "Association threshold < 0.2 captures noise",
        "error_high": "Association threshold > 0.8 leaves no room between tiers",
    },
    "birthright_promotion_threshold": {
        "min": 0.3,
        "max": 0.8,
        "error_low": "Birthright promotion threshold < 0.3 flags too many entitlements",
        "error_high": "Birthright promotion threshold > 0.8 misses cross-role entitlements",
    },
    "role_merge_similarity_threshold": {
        "min": 0.5,
        "max": 0.9,
        "error_low": "Merge threshold < 0.5 flags too many role pairs",
        "error_high": "Merge threshold > 0.9 misses near-identical roles",
    },
    "drift_auto_approve_threshold": {
        "min": 0.0,
        "max": 0.3,
        "error_high": "Auto-approve threshold > 0.3 too risky",
    },
}


# ============================================================================
# DEFAULT CONFIGURATION
# ============================================================================

DEFAULT_MINING_CONFIG: Dict[str, Any] = {
    # =========================================================================
    # BIRTHRIGHT DETECTION (Reused from V1)
    # =========================================================================
    "birthright_threshold": 0.8,  # 80%+ users = birthright
    "birthright_explicit": {
        # Force-include as birthright (dict of APP_ID -> [ENT_ID, ...])
        # Example: {"ActiveDirectory": ["User"], "Email": ["Access"]}
    },
    "min_assignment_count": 5,  # Filter out entitlements with < 5 assignments

    # =========================================================================
    # ENTITLEMENT CLUSTERING (Leiden-based, V2)
    # =========================================================================
    "entitlement_clustering_method": "leiden",  # or "hierarchical" (fallback)

    # Jaccard graph construction
    "leiden_min_similarity": 0.3,  # Jaccard threshold for edges (0.2-0.7)
    "leiden_min_shared_users": 3,  # Minimum users sharing entitlements for edge

    # Leiden algorithm parameters
    "leiden_resolution": 0.8,  # Controls granularity (0.5-2.0)
                              # Lower = fewer, larger clusters
                              # Higher = more, smaller clusters
    "leiden_random_seed": 42,  # For reproducibility

    # User assignment to clusters
    "min_entitlement_coverage": 0.5,  # User must have 50%+ of cluster entitlements
    "min_absolute_overlap": 2,            # User must match at least this many ents (floor)
    "max_clusters_per_user": 5,  # Cap multi-membership (realistic limit)

    # Cluster filtering
    "min_role_size": 10,  # Drop clusters with < 10 users
    "min_entitlements_per_role": 2,  # Drop clusters with < 2 entitlements

    # =========================================================================
    # BUSINESS ROLE MANAGEMENT (New for Hybrid)
    # =========================================================================
    # CHANGED: removed auto_generate_role_names, role_name_primary_attr,
    # role_name_secondary_attr, role_name_min_dominance.
    # Roles are named ROLE_001, ROLE_002 etc. Customer renames during review.

    # Role lifecycle
    "allow_draft_role_editing": True,  # Stakeholders can edit before approval
    "require_role_owner": True,  # Every business role must have owner
    "require_role_purpose": False,  # Purpose description optional

    # =========================================================================
    # DAILY RECLUSTERING (New for Hybrid)
    # =========================================================================
    "enable_daily_clustering": True,  # Turn on/off daily reclustering
    "daily_clustering_schedule": "0 2 * * *",  # Cron: 2am daily

    # Incremental clustering optimization
    "use_incremental_clustering": True,  # Faster if < 5% data change
    "incremental_threshold": 0.05,  # < 5% change = incremental update

    # Peer cluster retention
    "peer_cluster_retention_days": 90,  # Keep 90 days for trend analysis

    # =========================================================================
    # DRIFT DETECTION (New for Hybrid)
    # =========================================================================
    "enable_drift_detection": True,

    # Drift thresholds
    "drift_stable_threshold": 0.1,  # < 10% change = stable
    "drift_moderate_threshold": 0.25,  # 10-25% = moderate drift
    # > 25% = high drift

    # Entitlement drift
    "drift_new_entitlement_prevalence": 0.7,  # 70%+ cluster has it = recommend add
    "drift_declining_entitlement_prevalence": 0.3,  # < 30% = recommend remove

    # User drift
    "drift_user_movement_threshold": 0.1,  # > 10% users changed clusters = alert

    # Temporal analysis windows
    "drift_analysis_windows": [7, 30, 90],  # Days for trend analysis

    # =========================================================================
    # ROLE UPDATE WORKFLOW (New for Hybrid)
    # =========================================================================
    "enable_auto_approve": True,  # Allow auto-approval of low-drift changes

    # Auto-approval criteria (ALL must be met)
    "drift_auto_approve_threshold": 0.1,  # < 10% drift
    "drift_auto_approve_min_prevalence": 0.7,  # >= 70% prevalence
    "drift_auto_approve_min_stability_days": 3,  # Stable for 3+ days

    # High-drift requires stakeholder approval
    "drift_require_approval_threshold": 0.25,  # > 25% drift

    # Notification settings
    "notify_on_auto_approve": True,  # Email owner when auto-approved
    "notify_on_drift_alert": True,  # Email owner on high drift

    # =========================================================================
    # CONFIDENCE SCORING (Enhanced with Attributes + Drift, V2)
    # =========================================================================
    # CHANGED: removed use_attribute_weighting, attribute_weights dict,
    # attribute_columns dict. Replaced with customer-specified user_attributes.

    # Customer-specified user attributes for confidence scoring and HR summary.
    # Must be configured before mining. 2-5 attributes from identities.csv.
    # Each entry: {"column": "<col_name>", "weight": <0-1>} (weight optional).
    # If weights omitted, equal weight (1/N) assigned automatically.
    # If weights specified, must sum to 1.0.
    "user_attributes": [],  # REQUIRED - empty default, validation catches it

    # ADDED: prevalence-based role overlap thresholds
    "entitlement_prevalence_threshold": 0.75,  # >= 75% of role members = core entitlement
    "entitlement_association_threshold": 0.40,  # >= 40% and < prevalence = common entitlement
    "birthright_promotion_threshold": 0.50,     # core in > 50% of roles = flag for birthright
    "role_merge_similarity_threshold": 0.70,    # Jaccard > 0.70 of core sets = merge candidate

    # Peer group weight (base weight before attribute scaling)
    # Attribute weights are scaled into (1.0 - peer_group_weight - extra_factor_weights)
    "peer_group_weight": 0.40,

    # Drift stability factor (NEW)
    "use_drift_stability_factor": False,  # Stub returns 1.0 always - enable when drift_detector is implemented
    "drift_stability_weight": 0.0,  # Zero until drift_stability_factor is implemented
    "drift_stability_window_days": 7,  # Stable = unchanged for 7+ days

    # Role coverage factor (NEW)
    "use_role_coverage_factor": True,
    "role_coverage_weight": 0.1,  # Add 10% weight based on role coverage

    # Pre-computation settings
    "max_attribute_cardinality": 500,  # Skip attributes with > 500 unique values
    "renormalize_weights_on_null": True,  # Re-weight when attributes missing
    "min_attribute_group_size": 2,  # Skip attribute values with < 2 users

    # Confidence thresholds
    "confidence_high_threshold": 0.8,  # >= 80% = HIGH
    "confidence_medium_threshold": 0.5,  # 50-80% = MEDIUM, < 50% = LOW

    # Peer group reliability
    "min_reliable_peer_count": 5,  # Flag small peer groups as unreliable

    # =========================================================================
    # RECOMMENDATIONS & OVER-PROVISIONED (Reused from V1)
    # =========================================================================
    "recommendation_min_confidence": 0.6,  # Suggest if >= 60% confidence
    "revocation_threshold": 0.2,  # Flag for removal if < 20% confidence
    "min_user_assignments": 3,  # Skip users with < 3 non-birthright entitlements

    # =========================================================================
    # PERFORMANCE & OPTIMIZATION
    # =========================================================================
    "use_sparse_matrices": True,  # Required for 50K+ users
    "sparse_format": "csr",  # scipy.sparse format

    # Memory management
    "batch_size": 10000,  # For chunked operations
    "max_memory_mb": 2048,  # Stop if memory exceeds 2GB

    # Parallel processing (future)
    "parallel_attribute_computation": False,  # Multiprocessing for prevalence
    "num_workers": 4,  # CPU cores for parallel jobs

    # =========================================================================
    # OUTPUT & REPORTING
    # =========================================================================
    "include_coverage_in_exports": True,  # Add coverage columns to CSVs
    "generate_cluster_diagnostics": True,  # Detailed cluster metrics
    "save_intermediate_results": False,  # Debug: save graph, leiden output

    # Export formats
    "export_formats": ["csv", "json"],  # Available: csv, json, excel

    # =========================================================================
    # LOGGING & MONITORING
    # =========================================================================
    "log_level": "DEBUG",  # DEBUG, INFO, WARNING, ERROR
    "enable_performance_logging": True,  # Log timing for each step
    "enable_drift_logging": True,  # Log all drift events
}


# ============================================================================
# CONFIGURATION DATACLASS
# ============================================================================

@dataclass
class MiningConfig:
    """
    Strongly-typed configuration for Role Mining V2.

    Use this instead of dict for type safety and IDE autocomplete.
    """

    # Birthright
    birthright_threshold: float = 0.8
    birthright_explicit: Dict[str, List[str]] = field(default_factory=dict)
    min_assignment_count: int = 5

    # Leiden clustering
    entitlement_clustering_method: str = "leiden"
    leiden_min_similarity: float = 0.3
    leiden_min_shared_users: int = 3
    leiden_resolution: float = 1.0
    leiden_random_seed: int = 42
    min_entitlement_coverage: float = 0.5
    min_absolute_overlap: int = 2
    max_clusters_per_user: int = 5
    min_role_size: int = 10
    min_entitlements_per_role: int = 2

    # Business role management
    # CHANGED: removed auto_generate_role_names, role_name_primary_attr,
    # role_name_secondary_attr, role_name_min_dominance
    allow_draft_role_editing: bool = True
    require_role_owner: bool = True
    require_role_purpose: bool = False

    # Daily reclustering
    enable_daily_clustering: bool = True
    daily_clustering_schedule: str = "0 2 * * *"
    use_incremental_clustering: bool = True
    incremental_threshold: float = 0.05
    peer_cluster_retention_days: int = 90

    # Drift detection
    enable_drift_detection: bool = True
    drift_stable_threshold: float = 0.1
    drift_moderate_threshold: float = 0.25
    drift_new_entitlement_prevalence: float = 0.7
    drift_declining_entitlement_prevalence: float = 0.3
    drift_user_movement_threshold: float = 0.1
    drift_analysis_windows: List[int] = field(default_factory=lambda: [7, 30, 90])

    # Role update workflow
    enable_auto_approve: bool = True
    drift_auto_approve_threshold: float = 0.1
    drift_auto_approve_min_prevalence: float = 0.7
    drift_auto_approve_min_stability_days: int = 3
    drift_require_approval_threshold: float = 0.25
    notify_on_auto_approve: bool = True
    notify_on_drift_alert: bool = True

    # Confidence scoring
    # CHANGED: removed use_attribute_weighting, attribute_weights, attribute_columns
    # ADDED: user_attributes, prevalence thresholds
    user_attributes: List[Dict[str, Any]] = field(default_factory=list)
    entitlement_prevalence_threshold: float = 0.75
    entitlement_association_threshold: float = 0.40
    birthright_promotion_threshold: float = 0.50
    role_merge_similarity_threshold: float = 0.70
    use_drift_stability_factor: bool = False
    peer_group_weight: float = 0.40
    drift_stability_weight: float = 0.0
    drift_stability_window_days: int = 7
    use_role_coverage_factor: bool = True
    role_coverage_weight: float = 0.1
    max_attribute_cardinality: int = 500
    renormalize_weights_on_null: bool = True
    min_attribute_group_size: int = 2
    confidence_high_threshold: float = 0.8
    confidence_medium_threshold: float = 0.5
    min_reliable_peer_count: int = 5

    # Recommendations
    recommendation_min_confidence: float = 0.6
    revocation_threshold: float = 0.2
    min_user_assignments: int = 3

    # Performance
    use_sparse_matrices: bool = True
    sparse_format: str = "csr"
    batch_size: int = 10000
    max_memory_mb: int = 2048
    parallel_attribute_computation: bool = False
    num_workers: int = 4

    # Output
    include_coverage_in_exports: bool = True
    generate_cluster_diagnostics: bool = True
    save_intermediate_results: bool = False
    export_formats: List[str] = field(default_factory=lambda: ["csv", "json"])

    # Logging
    log_level: str = "INFO"
    enable_performance_logging: bool = True
    enable_drift_logging: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for backward compatibility."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MiningConfig':
        """Create from dictionary, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered)

    def validate(self) -> List[str]:
        """
        Validate configuration against rules.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # Leiden similarity
        rule = VALIDATION_RULES["leiden_min_similarity"]
        if self.leiden_min_similarity < rule["min"]:
            errors.append(f"leiden_min_similarity < {rule['min']}: {rule['error_low']}")
        if self.leiden_min_similarity > rule["max"]:
            errors.append(f"leiden_min_similarity > {rule['max']}: {rule['error_high']}")

        # Leiden min shared users
        rule = VALIDATION_RULES["leiden_min_shared_users"]
        if self.leiden_min_shared_users < rule["min"]:
            errors.append(f"leiden_min_shared_users < {rule['min']}: {rule['error']}")

        # Leiden resolution
        rule = VALIDATION_RULES["leiden_resolution"]
        if self.leiden_resolution < rule["min"]:
            errors.append(f"leiden_resolution < {rule['min']}: {rule.get('warning_low', 'Too low')}")
        if self.leiden_resolution > rule["max"]:
            errors.append(f"leiden_resolution > {rule['max']}: {rule.get('warning_high', 'Too high')}")

        # Coverage
        rule = VALIDATION_RULES["min_entitlement_coverage"]
        if self.min_entitlement_coverage < rule["min"]:
            errors.append(f"min_entitlement_coverage < {rule['min']}: {rule['error_low']}")
        if self.min_entitlement_coverage > rule["max"]:
            errors.append(f"min_entitlement_coverage > {rule['max']}: {rule['error_high']}")

        # CHANGED: replaced attribute_weights validation with user_attributes
        rule = VALIDATION_RULES["user_attributes"]
        if not isinstance(self.user_attributes, list):
            errors.append("user_attributes must be a list")
        elif len(self.user_attributes) < rule["min_count"] or len(self.user_attributes) > rule["max_count"]:
            errors.append(
                f"user_attributes has {len(self.user_attributes)} entries: {rule['error_count']}"
            )
        else:
            # Validate each entry has a column field
            columns_seen = set()
            has_any_weight = any("weight" in attr for attr in self.user_attributes)
            has_all_weights = all("weight" in attr for attr in self.user_attributes)

            for i, attr in enumerate(self.user_attributes):
                if not isinstance(attr, dict) or "column" not in attr:
                    errors.append(f"user_attributes[{i}]: {rule['error_missing_column']}")
                    continue

                col = attr["column"]
                if not isinstance(col, str) or not col.strip():
                    errors.append(f"user_attributes[{i}]: column must be a non-empty string")
                    continue

                if col in columns_seen:
                    errors.append(f"user_attributes[{i}]: duplicate column '{col}': {rule['error_duplicate']}")
                columns_seen.add(col)

            # Weight validation: either all have weights or none do
            if has_any_weight and not has_all_weights:
                errors.append("user_attributes: if any entry has a weight, all must have a weight")
            elif has_all_weights and len(self.user_attributes) >= rule["min_count"]:
                for i, attr in enumerate(self.user_attributes):
                    w = attr.get("weight", 0)
                    if not isinstance(w, (int, float)) or w <= 0 or w > 1.0:
                        errors.append(
                            f"user_attributes[{i}] weight={w}: {rule['error_weight_range']}"
                        )
                total_weight = sum(attr.get("weight", 0) for attr in self.user_attributes)
                if abs(total_weight - rule["weight_sum"]) > rule["weight_tolerance"]:
                    errors.append(
                        f"user_attributes weights sum to {total_weight:.3f}: {rule['error_weight_sum']}"
                    )

        # Max clusters per user
        rule = VALIDATION_RULES["max_clusters_per_user"]
        if self.max_clusters_per_user < rule["min"]:
            errors.append(f"max_clusters_per_user < {rule['min']}")
        if self.max_clusters_per_user > rule["max"]:
            errors.append(f"max_clusters_per_user > {rule['max']}: {rule.get('warning_high', 'Too many')}")

        # Min entitlements per role
        rule = VALIDATION_RULES["min_entitlements_per_role"]
        if self.min_entitlements_per_role < rule["min"]:
            errors.append(f"min_entitlements_per_role < {rule['min']}: {rule['error']}")

        # ADDED: prevalence threshold validations
        rule = VALIDATION_RULES["entitlement_prevalence_threshold"]
        if self.entitlement_prevalence_threshold < rule["min"]:
            errors.append(f"entitlement_prevalence_threshold < {rule['min']}: {rule['error_low']}")
        if self.entitlement_prevalence_threshold > rule["max"]:
            errors.append(f"entitlement_prevalence_threshold > {rule['max']}: {rule['error_high']}")

        rule = VALIDATION_RULES["entitlement_association_threshold"]
        if self.entitlement_association_threshold < rule["min"]:
            errors.append(f"entitlement_association_threshold < {rule['min']}: {rule['error_low']}")
        if self.entitlement_association_threshold > rule["max"]:
            errors.append(f"entitlement_association_threshold > {rule['max']}: {rule['error_high']}")

        # Cross-validation: association must be < prevalence
        if self.entitlement_association_threshold >= self.entitlement_prevalence_threshold:
            errors.append(
                f"entitlement_association_threshold ({self.entitlement_association_threshold}) "
                f"must be < entitlement_prevalence_threshold ({self.entitlement_prevalence_threshold})"
            )

        rule = VALIDATION_RULES["birthright_promotion_threshold"]
        if self.birthright_promotion_threshold < rule["min"]:
            errors.append(f"birthright_promotion_threshold < {rule['min']}: {rule['error_low']}")
        if self.birthright_promotion_threshold > rule["max"]:
            errors.append(f"birthright_promotion_threshold > {rule['max']}: {rule['error_high']}")

        rule = VALIDATION_RULES["role_merge_similarity_threshold"]
        if self.role_merge_similarity_threshold < rule["min"]:
            errors.append(f"role_merge_similarity_threshold < {rule['min']}: {rule['error_low']}")
        if self.role_merge_similarity_threshold > rule["max"]:
            errors.append(f"role_merge_similarity_threshold > {rule['max']}: {rule['error_high']}")

        # Drift auto-approve threshold
        rule = VALIDATION_RULES["drift_auto_approve_threshold"]
        if self.drift_auto_approve_threshold < rule["min"]:
            errors.append(f"drift_auto_approve_threshold < {rule['min']}")
        if self.drift_auto_approve_threshold > rule["max"]:
            errors.append(f"drift_auto_approve_threshold > {rule['max']}: {rule['error_high']}")

        return errors

    def validate_against_data(self, identities_columns: List[str]) -> List[str]:
        """
        ADDED: Validate that user_attributes columns exist in actual data.

        Called at mining time after file upload, not at config time.
        Hard fail if any configured attribute column is missing from identities.csv.

        Args:
            identities_columns: List of column names from identities.csv

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        # USR_ID is mandatory
        if "USR_ID" not in identities_columns:
            errors.append("identities.csv must contain 'USR_ID' column")

        # Every configured attribute column must exist
        for attr in self.user_attributes:
            col = attr.get("column", "")
            if col not in identities_columns:
                errors.append(
                    f"Attribute column '{col}' not found in identities.csv. "
                    f"Available columns: {', '.join(sorted(identities_columns))}"
                )

        return errors

    def get_attribute_columns(self) -> List[str]:
        """
        ADDED: Get list of configured attribute column names.

        Convenience method used by role_builder and confidence_scorer.
        """
        return [attr["column"] for attr in self.user_attributes]

    def get_attribute_weights(self) -> Dict[str, float]:
        """
        ADDED: Get column->weight mapping. Auto-assigns equal weights if not specified.

        Returns:
            Dict mapping column name to weight, guaranteed to sum to 1.0.
        """
        n = len(self.user_attributes)
        if n == 0:
            return {}

        has_weights = all("weight" in attr for attr in self.user_attributes)
        if has_weights:
            return {attr["column"]: attr["weight"] for attr in self.user_attributes}
        else:
            equal_weight = 1.0 / n
            return {attr["column"]: equal_weight for attr in self.user_attributes}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_config_from_json(filepath: str) -> MiningConfig:
    """Load configuration from JSON file."""
    with open(filepath, 'r') as f:
        config_dict = json.load(f)
    return MiningConfig.from_dict(config_dict)


def save_config_to_json(config: MiningConfig, filepath: str) -> None:
    """Save configuration to JSON file."""
    with open(filepath, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)


def get_default_config() -> MiningConfig:
    """Get default configuration as dataclass."""
    return MiningConfig.from_dict(DEFAULT_MINING_CONFIG)


def merge_configs(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge override config into base config.

    Useful for user-specific overrides on top of defaults.
    """
    merged = base.copy()
    merged.update(overrides)
    return merged


# ============================================================================
# CONFIGURATION PRESETS
# ============================================================================

# Conservative preset (fewer, larger roles)
CONSERVATIVE_CONFIG = {
    "leiden_min_similarity": 0.4,  # Higher = fewer edges
    "leiden_resolution": 0.8,  # Lower = larger clusters
    "min_entitlement_coverage": 0.6,  # Higher coverage required
    "max_clusters_per_user": 3,  # Limit multi-membership
    "drift_auto_approve_threshold": 0.05,  # Very low drift only
    "drift_auto_approve_min_prevalence": 0.8,  # 80%+ prevalence required
}

# Aggressive preset (more, smaller roles)
AGGRESSIVE_CONFIG = {
    "leiden_min_similarity": 0.25,  # Lower = more edges
    "leiden_resolution": 0.8,  # Higher = smaller clusters
    "min_entitlement_coverage": 0.4,  # Lower coverage accepted
    "max_clusters_per_user": 7,  # Allow more multi-membership
    "drift_auto_approve_threshold": 0.15,  # Higher drift auto-approved
    "drift_auto_approve_min_prevalence": 0.6,  # 60%+ prevalence sufficient
}

# Experimental preset (daily reclustering disabled, manual governance)
MANUAL_GOVERNANCE_CONFIG = {
    "enable_daily_clustering": False,
    "enable_drift_detection": False,
    "enable_auto_approve": False,
    "notify_on_drift_alert": False,
}


def get_preset_config(preset: str) -> Dict[str, Any]:
    """
    Get configuration preset.

    Args:
        preset: "default", "conservative", "aggressive", or "manual"

    Returns:
        Configuration dictionary
    """
    if preset == "conservative":
        return merge_configs(DEFAULT_MINING_CONFIG, CONSERVATIVE_CONFIG)
    elif preset == "aggressive":
        return merge_configs(DEFAULT_MINING_CONFIG, AGGRESSIVE_CONFIG)
    elif preset == "manual":
        return merge_configs(DEFAULT_MINING_CONFIG, MANUAL_GOVERNANCE_CONFIG)
    else:
        return DEFAULT_MINING_CONFIG.copy()


# ============================================================================
# SESSION-BASED FILE I/O (integrates with existing session.py)
# ============================================================================

def load_session_config(session_path: str) -> MiningConfig:
    """
    Load V2 configuration from session directory.

    Looks for config_v2.json, falls back to config.json, falls back to defaults.

    Args:
        session_path: Path to session directory (e.g., data/sessions/<session_id>)

    Returns:
        MiningConfigV2 instance
    """
    # Try V2-specific config first
    config_path = os.path.join(session_path, "config.json")
    if os.path.isfile(config_path):
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return MiningConfig.from_dict(config_dict)

    # No config found, use defaults
    return get_default_config()


def save_session_config(session_path: str, config: MiningConfig) -> None:
    """
    Save V2 configuration to session directory.

    Saves to config_v2.json (separate from V1 config.json).

    Args:
        session_path: Path to session directory
        config: Configuration to save
    """
    config_path = os.path.join(session_path, "config.json")
    with open(config_path, 'w') as f:
        json.dump(config.to_dict(), f, indent=2)


def get_results_path(session_path: str) -> str:
    """
    Get path to V2 results directory.

    V2 results stored separately: results_v2/
    V1 results remain in: results/

    Args:
        session_path: Path to session directory

    Returns:
        Path to results_v2 directory (creates if needed)
    """
    results_path = os.path.join(session_path, "results")
    os.makedirs(results_path, exist_ok=True)
    return results_path


def initialize_session_directories(session_path: str) -> None:
    """
    Create V2-specific directories in session.

    Creates:
        - results_v2/          (V2 mining results)
        - business_roles/      (approved business roles, versioned)
        - peer_clusters/       (daily peer cluster snapshots, last 90 days)
        - drift_reports/       (daily drift detection reports)

    Args:
        session_path: Path to session directory
    """
    os.makedirs(os.path.join(session_path, "results"), exist_ok=True)
    os.makedirs(os.path.join(session_path, "business_roles"), exist_ok=True)
    os.makedirs(os.path.join(session_path, "peer_clusters"), exist_ok=True)
    os.makedirs(os.path.join(session_path, "drift_reports"), exist_ok=True)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example 1: Use default config
    config = get_default_config()
    errors = config.validate()
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Configuration valid ✓")

    # Example 2: Session-based I/O (simulated)
    # session_path = "data/sessions/abc-123"
    # config = load_session_config_v2(session_path)
    # save_session_config_v2(session_path, config)

    # Example 3: Use preset
    conservative = MiningConfig.from_dict(get_preset_config("conservative"))
    print(f"\nConservative preset:")
    print(f"  - min_similarity: {conservative.leiden_min_similarity}")
    print(f"  - resolution: {conservative.leiden_resolution}")
    print(f"  - coverage: {conservative.min_entitlement_coverage}")

    # Example 4: Override specific values
    custom_config = get_default_config()
    custom_config.leiden_resolution = 1.2
    custom_config.max_clusters_per_user = 4

    # Example 5: Validate before use
    errors = custom_config.validate()
    if not errors:
        print("\n✓ Custom config valid")

    print("\nConfiguration module loaded successfully.")