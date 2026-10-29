"""
data_processing.py
------------------
Handles synthetic data generation, CSV loading, validation,
cleaning, and feature/label separation for Cyber Guardian AI.

All functions return a (result, error_message) tuple:
  - On success: (result, None)
  - On failure: (None, descriptive_error_string)
"""

from __future__ import annotations

import io
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABEL_COLUMN = "label"
BENIGN_LABEL = "BENIGN"
ATTACK_LABELS = [
    "DoS",
    "DDoS",
    "PortScan",
    "BruteForce",
    "Web Attack",
    "Infiltration",
    "Botnet",
]

REQUIRED_FEATURE_COLUMNS = [
    "duration",
    "protocol_type",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "count",
    "srv_count",
    "same_srv_rate",
    "diff_srv_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
]

NUMERIC_FEATURES = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "count",
    "srv_count",
    "same_srv_rate",
    "diff_srv_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
]

CATEGORICAL_FEATURES = ["protocol_type"]

MIN_SAMPLES_FOR_TRAINING = 20
MIN_SAMPLES_PER_CLASS = 5


# ---------------------------------------------------------------------------
# Synthetic Data Generation
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(
    n_benign: int = 600,
    n_attack: int = 400,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Generate a clearly-labelled synthetic network-traffic dataset for
    demonstration purposes.

    The generated data is NOT real network traffic. Statistical
    distributions are designed to make the classification task learnable
    while remaining distinct from any real-world benchmark.

    Parameters
    ----------
    n_benign : int
        Number of benign records to generate.
    n_attack : int
        Number of attack records to generate (distributed across attack types).
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Shuffled DataFrame with feature columns and a ``label`` column.
    """
    rng = np.random.default_rng(random_state)

    protocols = ["tcp", "udp", "icmp"]

    def _benign_records(n: int) -> pd.DataFrame:
        return pd.DataFrame({
            "duration": rng.exponential(scale=10, size=n).clip(0, 300),
            "protocol_type": rng.choice(protocols, size=n, p=[0.7, 0.2, 0.1]),
            "src_bytes": rng.integers(100, 50_000, size=n).astype(float),
            "dst_bytes": rng.integers(100, 100_000, size=n).astype(float),
            "land": rng.choice([0, 1], size=n, p=[0.99, 0.01]),
            "wrong_fragment": rng.choice([0, 1, 2], size=n, p=[0.95, 0.04, 0.01]),
            "urgent": rng.choice([0, 1], size=n, p=[0.999, 0.001]),
            "hot": rng.integers(0, 5, size=n).astype(float),
            "num_failed_logins": rng.choice([0, 1], size=n, p=[0.97, 0.03]),
            "logged_in": rng.choice([0, 1], size=n, p=[0.4, 0.6]),
            "num_compromised": rng.choice([0, 1, 2], size=n, p=[0.97, 0.02, 0.01]),
            "count": rng.integers(1, 100, size=n).astype(float),
            "srv_count": rng.integers(1, 100, size=n).astype(float),
            "same_srv_rate": rng.uniform(0.5, 1.0, size=n),
            "diff_srv_rate": rng.uniform(0.0, 0.3, size=n),
            "dst_host_count": rng.integers(1, 255, size=n).astype(float),
            "dst_host_srv_count": rng.integers(1, 255, size=n).astype(float),
            "dst_host_same_srv_rate": rng.uniform(0.5, 1.0, size=n),
            "dst_host_diff_srv_rate": rng.uniform(0.0, 0.3, size=n),
            "dst_host_same_src_port_rate": rng.uniform(0.0, 0.5, size=n),
            "label": BENIGN_LABEL,
        })

    def _dos_records(n: int) -> pd.DataFrame:
        return pd.DataFrame({
            "duration": rng.uniform(0, 0.5, size=n),
            "protocol_type": rng.choice(protocols, size=n, p=[0.1, 0.6, 0.3]),
            "src_bytes": rng.integers(0, 500, size=n).astype(float),
            "dst_bytes": rng.integers(0, 100, size=n).astype(float),
            "land": rng.choice([0, 1], size=n, p=[0.8, 0.2]),
            "wrong_fragment": rng.integers(0, 3, size=n).astype(float),
            "urgent": rng.choice([0, 1], size=n, p=[0.9, 0.1]),
            "hot": rng.integers(0, 2, size=n).astype(float),
            "num_failed_logins": rng.choice([0, 1], size=n, p=[0.9, 0.1]),
            "logged_in": rng.choice([0, 1], size=n, p=[0.9, 0.1]),
            "num_compromised": rng.integers(0, 2, size=n).astype(float),
            "count": rng.integers(200, 511, size=n).astype(float),
            "srv_count": rng.integers(200, 511, size=n).astype(float),
            "same_srv_rate": rng.uniform(0.8, 1.0, size=n),
            "diff_srv_rate": rng.uniform(0.0, 0.1, size=n),
            "dst_host_count": rng.integers(200, 255, size=n).astype(float),
            "dst_host_srv_count": rng.integers(200, 255, size=n).astype(float),
            "dst_host_same_srv_rate": rng.uniform(0.8, 1.0, size=n),
            "dst_host_diff_srv_rate": rng.uniform(0.0, 0.1, size=n),
            "dst_host_same_src_port_rate": rng.uniform(0.0, 0.1, size=n),
            "label": "DoS",
        })

    def _portscan_records(n: int) -> pd.DataFrame:
        return pd.DataFrame({
            "duration": rng.uniform(0, 1, size=n),
            "protocol_type": rng.choice(protocols, size=n, p=[0.8, 0.15, 0.05]),
            "src_bytes": rng.integers(0, 200, size=n).astype(float),
            "dst_bytes": rng.integers(0, 200, size=n).astype(float),
            "land": np.zeros(n),
            "wrong_fragment": np.zeros(n),
            "urgent": np.zeros(n),
            "hot": np.zeros(n),
            "num_failed_logins": np.zeros(n),
            "logged_in": np.zeros(n),
            "num_compromised": np.zeros(n),
            "count": rng.integers(1, 20, size=n).astype(float),
            "srv_count": rng.integers(1, 5, size=n).astype(float),
            "same_srv_rate": rng.uniform(0.0, 0.2, size=n),
            "diff_srv_rate": rng.uniform(0.7, 1.0, size=n),
            "dst_host_count": rng.integers(1, 50, size=n).astype(float),
            "dst_host_srv_count": rng.integers(1, 10, size=n).astype(float),
            "dst_host_same_srv_rate": rng.uniform(0.0, 0.1, size=n),
            "dst_host_diff_srv_rate": rng.uniform(0.8, 1.0, size=n),
            "dst_host_same_src_port_rate": rng.uniform(0.9, 1.0, size=n),
            "label": "PortScan",
        })

    def _bruteforce_records(n: int) -> pd.DataFrame:
        return pd.DataFrame({
            "duration": rng.uniform(0.5, 5, size=n),
            "protocol_type": rng.choice(protocols, size=n, p=[0.95, 0.04, 0.01]),
            "src_bytes": rng.integers(500, 5000, size=n).astype(float),
            "dst_bytes": rng.integers(500, 5000, size=n).astype(float),
            "land": np.zeros(n),
            "wrong_fragment": np.zeros(n),
            "urgent": np.zeros(n),
            "hot": rng.integers(0, 3, size=n).astype(float),
            "num_failed_logins": rng.integers(3, 10, size=n).astype(float),
            "logged_in": rng.choice([0, 1], size=n, p=[0.7, 0.3]),
            "num_compromised": rng.integers(0, 5, size=n).astype(float),
            "count": rng.integers(10, 100, size=n).astype(float),
            "srv_count": rng.integers(1, 10, size=n).astype(float),
            "same_srv_rate": rng.uniform(0.8, 1.0, size=n),
            "diff_srv_rate": rng.uniform(0.0, 0.2, size=n),
            "dst_host_count": rng.integers(1, 30, size=n).astype(float),
            "dst_host_srv_count": rng.integers(1, 30, size=n).astype(float),
            "dst_host_same_srv_rate": rng.uniform(0.7, 1.0, size=n),
            "dst_host_diff_srv_rate": rng.uniform(0.0, 0.2, size=n),
            "dst_host_same_src_port_rate": rng.uniform(0.5, 1.0, size=n),
            "label": "BruteForce",
        })

    # Distribute attack records across 4 types
    n_per_type = n_attack // 4
    remainder = n_attack - n_per_type * 4

    frames = [
        _benign_records(n_benign),
        _dos_records(n_per_type + remainder),
        _portscan_records(n_per_type),
        _bruteforce_records(n_per_type),
        _dos_records(n_per_type),   # extra DDoS-like
    ]
    # Relabel last frame
    frames[-1] = frames[-1].copy()
    frames[-1]["label"] = "DDoS"

    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class ValidationError(Exception):
    """Raised when the dataset fails validation checks."""


def validate_dataframe(
    df: pd.DataFrame,
    require_label: bool = True,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Validate a DataFrame for use in model training or prediction.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate.
    require_label : bool
        Whether the ``label`` column must be present.

    Returns
    -------
    (DataFrame, None) on success, or (None, error_message) on failure.
    """
    if df is None or df.empty:
        return None, "The dataset is empty. Please upload a file with records."

    if len(df) < 2:
        return None, f"Dataset has only {len(df)} record(s). At least 2 are needed."

    missing_features = [c for c in REQUIRED_FEATURE_COLUMNS if c not in df.columns]
    if missing_features:
        return None, (
            f"Missing required feature columns: {missing_features}. "
            "Please check the expected CSV format in the Data Explorer."
        )

    if require_label and LABEL_COLUMN not in df.columns:
        return None, (
            f"Column '{LABEL_COLUMN}' is required for training but was not found. "
            "For prediction-only data, use the prediction mode."
        )

    if require_label:
        unique_labels = df[LABEL_COLUMN].dropna().unique()
        if len(unique_labels) < 2:
            return None, (
                f"Training requires at least 2 distinct labels. "
                f"Found: {list(unique_labels)}."
            )

        label_counts = df[LABEL_COLUMN].value_counts()
        small_classes = label_counts[label_counts < MIN_SAMPLES_PER_CLASS]
        if not small_classes.empty:
            return None, (
                f"Some classes have too few samples for a reliable split: "
                f"{small_classes.to_dict()}. Each class needs at least "
                f"{MIN_SAMPLES_PER_CLASS} samples."
            )

        if len(df) < MIN_SAMPLES_FOR_TRAINING:
            return None, (
                f"Dataset has only {len(df)} records. "
                f"At least {MIN_SAMPLES_FOR_TRAINING} are needed for training."
            )

    return df, None


# ---------------------------------------------------------------------------
# Loading and Preprocessing
# ---------------------------------------------------------------------------

def load_csv_bytes(file_bytes: bytes) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Parse a CSV from raw bytes (uploaded via Streamlit).

    Returns (DataFrame, None) on success or (None, error) on failure.
    """
    try:
        df = pd.read_csv(io.BytesIO(file_bytes))
    except Exception as exc:
        return None, f"Could not parse the CSV file: {exc}"

    if df.empty:
        return None, "The uploaded CSV file contains no data rows."

    # Strip leading/trailing whitespace from column names
    df.columns = df.columns.str.strip()

    return df, None


def preprocess_dataframe(
    df: pd.DataFrame,
    require_label: bool = True,
) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """
    Clean and prepare a DataFrame for the ML pipeline.

    Steps:
    1. Drop irrelevant index-like columns.
    2. Strip whitespace from string columns.
    3. Coerce numeric columns.
    4. Replace non-finite values with NaN, then fill with median/mode.
    5. Validate with :func:`validate_dataframe`.

    Returns (cleaned_df, None) or (None, error_message).
    """
    df = df.copy()

    # Drop likely index columns
    index_like = [c for c in df.columns if c.lower() in ("index", "id", "row", "no", "#")]
    df = df.drop(columns=index_like, errors="ignore")

    # Strip string whitespace in object columns
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip()

    # Coerce numeric feature columns
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Replace non-finite floats with NaN
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)

    # Fill missing values
    for col in df.select_dtypes(include=[np.number]).columns:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include=["object", "string"]).columns:
        if col == LABEL_COLUMN:
            continue
        if df[col].isna().any():
            mode_val = df[col].mode()
            df[col] = df[col].fillna(mode_val[0] if not mode_val.empty else "unknown")

    validated_df, err = validate_dataframe(df, require_label=require_label)
    return validated_df, err


def get_features_and_labels(
    df: pd.DataFrame,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series], Optional[str]]:
    """
    Separate features (X) and labels (y) from a preprocessed DataFrame.

    Returns (X, y, None) on success or (None, None, error) on failure.
    """
    if LABEL_COLUMN not in df.columns:
        return None, None, f"Column '{LABEL_COLUMN}' not found."

    available_features = [c for c in REQUIRED_FEATURE_COLUMNS if c in df.columns]
    if not available_features:
        return None, None, "No usable feature columns found."

    X = df[available_features].copy()
    y = df[LABEL_COLUMN].copy()

    return X, y, None


def get_class_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame summarising label counts and percentages."""
    if LABEL_COLUMN not in df.columns:
        return pd.DataFrame(columns=["Label", "Count", "Percentage"])

    counts = df[LABEL_COLUMN].value_counts()
    total = counts.sum()
    result = pd.DataFrame({
        "Label": counts.index,
        "Count": counts.values,
        "Percentage": (counts.values / total * 100).round(2),
    })
    return result


def get_dataset_summary(df: pd.DataFrame) -> dict:
    """Return a summary dict of dataset statistics."""
    summary = {
        "total_records": len(df),
        "total_columns": len(df.columns),
        "has_label": LABEL_COLUMN in df.columns,
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "numeric_columns": list(df.select_dtypes(include=np.number).columns),
        "categorical_columns": list(df.select_dtypes(include=["object", "string"]).columns),
    }
    if summary["has_label"]:
        summary["class_distribution"] = df[LABEL_COLUMN].value_counts().to_dict()
        summary["num_classes"] = df[LABEL_COLUMN].nunique()
    return summary

