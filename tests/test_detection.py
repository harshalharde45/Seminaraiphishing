"""
test_detection.py
-----------------
Integration tests for src/detection.py — end-to-end workflow.
"""
import numpy as np
import pandas as pd
import pytest

from src.detection import (
    EvaluationResult,
    PredictionResult,
    TrainingResult,
    get_threat_summary,
    run_evaluation,
    run_prediction,
    run_training,
)
from src.data_processing import LABEL_COLUMN, generate_synthetic_dataset


@pytest.fixture(scope="module")
def trained_state(synthetic_df):
    """Train once for the whole module, share the result."""
    result = run_training(synthetic_df)
    return result


class TestRunTraining:

    def test_training_succeeds(self, trained_state):
        assert trained_state.success is True
        assert trained_state.pipeline is not None
        assert trained_state.X_test is not None
        assert trained_state.y_test is not None

    def test_train_and_test_splits_are_sensible(self, trained_state, synthetic_df):
        total = len(synthetic_df)
        assert trained_state.train_samples > 0
        assert trained_state.test_samples > 0
        assert trained_state.train_samples + trained_state.test_samples == pytest.approx(
            total, abs=5
        )

    def test_classes_populated(self, trained_state):
        assert len(trained_state.classes) >= 2

    def test_model_info_present(self, trained_state):
        assert "model_type" in trained_state.model_info

    def test_empty_df_fails(self):
        result = run_training(pd.DataFrame())
        assert result.success is False
        assert result.error is not None

    def test_unlabelled_df_fails(self, synthetic_df):
        df_no_label = synthetic_df.drop(columns=[LABEL_COLUMN])
        result = run_training(df_no_label)
        assert result.success is False


class TestRunEvaluation:

    def test_evaluation_succeeds(self, trained_state):
        ev = run_evaluation(
            trained_state.pipeline,
            trained_state.X_test,
            trained_state.y_test,
        )
        assert ev.success is True
        assert ev.metrics is not None

    def test_metrics_keys_present(self, trained_state):
        ev = run_evaluation(
            trained_state.pipeline,
            trained_state.X_test,
            trained_state.y_test,
        )
        for key in ("accuracy", "f1_macro", "precision_macro", "recall_macro"):
            assert key in ev.metrics

    def test_accuracy_in_range(self, trained_state):
        ev = run_evaluation(
            trained_state.pipeline,
            trained_state.X_test,
            trained_state.y_test,
        )
        assert 0.0 <= ev.metrics["accuracy"] <= 1.0

    def test_per_class_df_not_empty(self, trained_state):
        ev = run_evaluation(
            trained_state.pipeline,
            trained_state.X_test,
            trained_state.y_test,
        )
        assert ev.per_class_df is not None
        assert not ev.per_class_df.empty

    def test_feature_importance_present(self, trained_state):
        ev = run_evaluation(
            trained_state.pipeline,
            trained_state.X_test,
            trained_state.y_test,
        )
        assert ev.feature_importance is not None
        assert len(ev.feature_importance) > 0


class TestRunPrediction:

    def test_prediction_on_unlabelled_data(self, trained_state, feature_only_df):
        result = run_prediction(trained_state.pipeline, feature_only_df)
        assert result.success is True
        assert result.predictions is not None
        assert len(result.predictions) == len(feature_only_df)

    def test_result_df_has_prediction_columns(self, trained_state, feature_only_df):
        result = run_prediction(trained_state.pipeline, feature_only_df)
        assert "predicted_label" in result.result_df.columns
        assert "risk_level" in result.result_df.columns
        assert "confidence_score" in result.result_df.columns

    def test_risk_level_values(self, trained_state, feature_only_df):
        result = run_prediction(trained_state.pipeline, feature_only_df)
        assert set(result.result_df["risk_level"].unique()).issubset({"HIGH", "LOW"})

    def test_confidence_score_in_range(self, trained_state, feature_only_df):
        result = run_prediction(trained_state.pipeline, feature_only_df)
        scores = result.result_df["confidence_score"]
        assert (scores >= 0.0).all()
        assert (scores <= 1.0).all()

    def test_empty_df_fails_gracefully(self, trained_state):
        result = run_prediction(trained_state.pipeline, pd.DataFrame())
        assert result.success is False
        assert result.error is not None


class TestThreatSummary:

    def test_summary_from_prediction_result(self, trained_state, feature_only_df):
        result = run_prediction(trained_state.pipeline, feature_only_df)
        summary = get_threat_summary(result.result_df)
        assert "total" in summary
        assert "benign_count" in summary
        assert "attack_count" in summary
        assert summary["benign_count"] + summary["attack_count"] == summary["total"]

    def test_summary_empty_df(self):
        summary = get_threat_summary(pd.DataFrame())
        assert summary == {}

    def test_percentage_sums_to_100(self, trained_state, feature_only_df):
        result = run_prediction(trained_state.pipeline, feature_only_df)
        summary = get_threat_summary(result.result_df)
        total_pct = summary["benign_pct"] + summary["attack_pct"]
        assert abs(total_pct - 100.0) < 0.5

