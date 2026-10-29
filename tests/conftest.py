"""
conftest.py — shared pytest fixtures for Cyber Guardian AI tests.
"""
import sys
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
import pandas as pd

from src.data_processing import (
    LABEL_COLUMN,
    REQUIRED_FEATURE_COLUMNS,
    generate_synthetic_dataset,
)


@pytest.fixture(scope="session")
def synthetic_df():
    """Full synthetic dataset for integration-level tests."""
    return generate_synthetic_dataset(n_benign=200, n_attack=100, random_state=42)


@pytest.fixture(scope="session")
def small_valid_df():
    """Minimal valid labelled DataFrame with all required columns."""
    rng = np.random.default_rng(0)
    n = 40
    data = {col: rng.uniform(0, 1, size=n) for col in REQUIRED_FEATURE_COLUMNS if col != "protocol_type"}
    data["protocol_type"] = np.where(rng.random(n) > 0.5, "tcp", "udp")
    data[LABEL_COLUMN] = np.where(rng.random(n) > 0.5, "BENIGN", "DoS")
    return pd.DataFrame(data)


@pytest.fixture(scope="session")
def feature_only_df(small_valid_df):
    """DataFrame without a label column (for prediction tests)."""
    return small_valid_df.drop(columns=[LABEL_COLUMN])

