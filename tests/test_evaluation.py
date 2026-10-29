"""
test_evaluation.py
------------------
Tests for src/evaluation.py — metric computation and interpretation.
"""
import numpy as np
import pandas as pd
import pytest

from src.evaluation import (
    compute_metrics,
    format_fp_fn_table,
    format_per_class_metrics,
    interpret_metrics,
)


class TestComputeMetrics:

    def test_perfect_classifier(self):
        y_true = np.array(["BENIGN", "DoS", "BENIGN", "DoS"])
        y_pred = np.array(["BENIGN", "DoS", "BENIGN", "DoS"])
        metrics, err = compute_metrics(y_true, y_pred)
        assert err is None
        assert metrics["accuracy"] == pytest.approx(1.0)
        assert metrics["f1_macro"] == pytest.approx(1.0)

    def test_worst_classifier(self):
        y_true = np.array(["BENIGN"] * 10 + ["DoS"] * 10)
        y_pred = np.array(["DoS"] * 10 + ["BENIGN"] * 10)
        metrics, err = compute_metrics(y_true, y_pred)
        assert err is None
        assert metrics["accuracy"] == pytest.approx(0.0)

    def test_multiclass_metrics(self):
        y_true = np.array(["BENIGN", "DoS", "PortScan", "BENIGN", "DoS", "PortScan"])
        y_pred = np.array(["BENIGN", "DoS", "BENIGN", "BENIGN", "PortScan", "PortScan"])
        metrics, err = compute_metrics(y_true, y_pred)
        assert err is None
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert "classes" in metrics
        assert len(metrics["classes"]) == 3

    def test_empty_y_true_returns_error(self):
        metrics, err = compute_metrics(np.array([]), np.array([]))
        assert err is not None
        assert metrics is None

    def test_mismatched_lengths_return_error(self):
        metrics, err = compute_metrics(
            np.array(["BENIGN", "DoS"]),
            np.array(["BENIGN"])
        )
        assert err is not None

    def test_confusion_matrix_present(self):
        y_true = np.array(["BENIGN", "DoS", "BENIGN"])
        y_pred = np.array(["BENIGN", "BENIGN", "DoS"])
        metrics, err = compute_metrics(y_true, y_pred)
        assert err is None
        assert "confusion_matrix" in metrics
        assert metrics["confusion_matrix"].shape == (2, 2)

    def test_fp_fn_present(self):
        y_true = np.array(["BENIGN", "DoS", "BENIGN", "DoS"])
        y_pred = np.array(["BENIGN", "BENIGN", "BENIGN", "DoS"])
        metrics, err = compute_metrics(y_true, y_pred)
        assert err is None
        assert "fp_fn_per_class" in metrics
        assert len(metrics["fp_fn_per_class"]) == 2

    def test_fn_count_correct(self):
        """DoS record predicted as BENIGN → 1 FN for DoS."""
        y_true = np.array(["BENIGN", "DoS", "BENIGN", "DoS"])
        y_pred = np.array(["BENIGN", "BENIGN", "BENIGN", "DoS"])
        metrics, _ = compute_metrics(y_true, y_pred)
        dos_row = next(r for r in metrics["fp_fn_per_class"] if r["class"] == "DoS")
        assert dos_row["FN"] == 1

    def test_metrics_all_between_zero_and_one(self):
        y_true = np.array(["BENIGN", "DoS"] * 10)
        y_pred = np.array(["DoS", "BENIGN"] * 5 + ["BENIGN", "DoS"] * 5)
        metrics, err = compute_metrics(y_true, y_pred)
        assert err is None
        for key in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"):
            assert 0.0 <= metrics[key] <= 1.0, f"{key} out of range: {metrics[key]}"

    def test_single_class_warning(self):
        """Single class in y_true should produce a warning."""
        y_true = np.array(["BENIGN"] * 10)
        y_pred = np.array(["BENIGN"] * 10)
        metrics, err = compute_metrics(y_true, y_pred)
        assert err is None
        assert len(metrics.get("warnings", [])) > 0


class TestFormattingFunctions:

    @pytest.fixture
    def sample_metrics(self):
        y_true = np.array(["BENIGN"] * 20 + ["DoS"] * 15 + ["PortScan"] * 10)
        y_pred = np.array(["BENIGN"] * 18 + ["DoS"] * 2 + ["DoS"] * 14 + ["PortScan"] * 1 + ["DoS"] * 3 + ["PortScan"] * 7)
        metrics, _ = compute_metrics(y_true, y_pred)
        return metrics

    def test_per_class_df_has_expected_columns(self, sample_metrics):
        df = format_per_class_metrics(sample_metrics)
        assert "Class" in df.columns
        assert "Precision" in df.columns
        assert "Recall" in df.columns
        assert "F1-Score" in df.columns
        assert "Support" in df.columns

    def test_fp_fn_table_has_expected_columns(self, sample_metrics):
        df = format_fp_fn_table(sample_metrics)
        assert "class" in df.columns
        assert "TP" in df.columns
        assert "FP" in df.columns
        assert "FN" in df.columns
        assert "TN" in df.columns

    def test_interpret_metrics_returns_list(self, sample_metrics):
        observations = interpret_metrics(sample_metrics)
        assert isinstance(observations, list)
        assert len(observations) > 0
        for obs in observations:
            assert isinstance(obs, str)

