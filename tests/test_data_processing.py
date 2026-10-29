"""
test_data_processing.py
-----------------------
Tests for src/data_processing.py — validation, loading, and preprocessing.
"""
import io
import numpy as np
import pandas as pd
import pytest

from src.data_processing import (
    LABEL_COLUMN,
    REQUIRED_FEATURE_COLUMNS,
    generate_synthetic_dataset,
    get_class_distribution,
    get_dataset_summary,
    load_csv_bytes,
    preprocess_dataframe,
    validate_dataframe,
)


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

class TestSyntheticDataGeneration:

    def test_generates_expected_columns(self):
        df = generate_synthetic_dataset(n_benign=100, n_attack=50)
        for col in REQUIRED_FEATURE_COLUMNS:
            assert col in df.columns, f"Missing column: {col}"
        assert LABEL_COLUMN in df.columns

    def test_generates_expected_row_count(self):
        df = generate_synthetic_dataset(n_benign=100, n_attack=80)
        # n_benign + attack rows (distributed with extras) >= n_benign + n_attack
        assert len(df) >= 100 + 80

    def test_generates_multiple_classes(self):
        df = generate_synthetic_dataset(n_benign=100, n_attack=100)
        unique_labels = df[LABEL_COLUMN].unique()
        assert len(unique_labels) >= 2

    def test_benign_class_present(self):
        df = generate_synthetic_dataset(n_benign=50, n_attack=50)
        assert "BENIGN" in df[LABEL_COLUMN].values

    def test_reproducible_with_same_seed(self):
        df1 = generate_synthetic_dataset(random_state=42)
        df2 = generate_synthetic_dataset(random_state=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_produce_different_data(self):
        df1 = generate_synthetic_dataset(random_state=1)
        df2 = generate_synthetic_dataset(random_state=2)
        assert not df1["src_bytes"].equals(df2["src_bytes"])

    def test_no_null_values(self):
        df = generate_synthetic_dataset(n_benign=100, n_attack=100)
        assert df.isnull().sum().sum() == 0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

class TestValidation:

    def test_valid_df_passes(self, small_valid_df):
        result, err = validate_dataframe(small_valid_df, require_label=True)
        assert err is None
        assert result is not None

    def test_empty_df_rejected(self):
        empty_df = pd.DataFrame()
        result, err = validate_dataframe(empty_df)
        assert result is None
        assert err is not None
        assert "empty" in err.lower()

    def test_none_rejected(self):
        result, err = validate_dataframe(None)
        assert result is None
        assert err is not None

    def test_missing_feature_columns_rejected(self):
        df = pd.DataFrame({"duration": [1, 2, 3], LABEL_COLUMN: ["BENIGN"] * 3})
        result, err = validate_dataframe(df, require_label=True)
        assert result is None
        assert "missing" in err.lower()

    def test_missing_label_column_rejected_when_required(self, feature_only_df):
        result, err = validate_dataframe(feature_only_df, require_label=True)
        assert result is None
        assert "label" in err.lower() or LABEL_COLUMN in err

    def test_missing_label_column_ok_when_not_required(self, feature_only_df):
        result, err = validate_dataframe(feature_only_df, require_label=False)
        assert err is None

    def test_single_class_rejected_for_training(self, small_valid_df):
        df_one_class = small_valid_df.copy()
        df_one_class[LABEL_COLUMN] = "BENIGN"
        result, err = validate_dataframe(df_one_class, require_label=True)
        assert result is None
        assert err is not None

    def test_insufficient_samples_rejected(self):
        rng = np.random.default_rng(0)
        n = 3
        data = {col: rng.uniform(0, 1, size=n) for col in REQUIRED_FEATURE_COLUMNS if col != "protocol_type"}
        data["protocol_type"] = ["tcp"] * n
        data[LABEL_COLUMN] = ["BENIGN", "DoS", "BENIGN"]
        df_tiny = pd.DataFrame(data)
        result, err = validate_dataframe(df_tiny, require_label=True)
        assert result is None or err is not None  # Either tiny or class-size failure


# ---------------------------------------------------------------------------
# CSV Loading
# ---------------------------------------------------------------------------

class TestCSVLoading:

    def test_valid_csv_loads(self, small_valid_df):
        csv_bytes = small_valid_df.to_csv(index=False).encode()
        result, err = load_csv_bytes(csv_bytes)
        assert err is None
        assert result is not None
        assert len(result) == len(small_valid_df)

    def test_invalid_bytes_rejected(self):
        result, err = load_csv_bytes(b"\x00\x01\x02\xff\xfe binary garbage")
        # Should either return an error or an empty-ish df — not crash
        assert result is None or err is not None or result.empty

    def test_empty_csv_rejected(self):
        result, err = load_csv_bytes(b"col1,col2\n")
        assert result is None or (err is not None) or result.empty

    def test_column_whitespace_stripped(self):
        csv_bytes = b"  duration  , protocol_type , label\n1,tcp,BENIGN\n"
        result, err = load_csv_bytes(csv_bytes)
        if result is not None:
            assert "duration" in result.columns
            assert "protocol_type" in result.columns


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

class TestPreprocessing:

    def test_valid_df_preprocesses(self, small_valid_df):
        result, err = preprocess_dataframe(small_valid_df, require_label=True)
        assert err is None
        assert result is not None

    def test_fills_missing_numeric_values(self, small_valid_df):
        df_with_nulls = small_valid_df.copy()
        df_with_nulls.loc[0, "duration"] = np.nan
        result, err = preprocess_dataframe(df_with_nulls, require_label=True)
        assert err is None
        assert not result["duration"].isnull().any()

    def test_fills_inf_values(self, small_valid_df):
        df_with_inf = small_valid_df.copy()
        df_with_inf.loc[0, "src_bytes"] = np.inf
        result, err = preprocess_dataframe(df_with_inf, require_label=True)
        assert err is None
        assert not np.isinf(result["src_bytes"]).any()

    def test_coerces_non_numeric(self, small_valid_df):
        df_str = small_valid_df.copy()
        df_str["src_bytes"] = df_str["src_bytes"].astype(str)
        df_str.loc[0, "src_bytes"] = "not_a_number"
        result, err = preprocess_dataframe(df_str, require_label=True)
        assert err is None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

class TestUtilities:

    def test_class_distribution(self, synthetic_df):
        dist = get_class_distribution(synthetic_df)
        assert "Label" in dist.columns
        assert "Count" in dist.columns
        assert "Percentage" in dist.columns
        assert dist["Percentage"].sum() == pytest.approx(100.0, abs=0.1)

    def test_dataset_summary(self, synthetic_df):
        summary = get_dataset_summary(synthetic_df)
        assert summary["total_records"] == len(synthetic_df)
        assert summary["has_label"] is True
        assert "class_distribution" in summary

    def test_class_distribution_no_label(self, feature_only_df):
        dist = get_class_distribution(feature_only_df)
        assert dist.empty or "Label" in dist.columns

