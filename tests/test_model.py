"""
test_model.py
-------------
Tests for src/model.py — pipeline building, training, and prediction.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.model import (
    build_pipeline,
    get_feature_importance,
    get_model_info,
    predict,
    train_model,
)
from src.data_processing import (
    LABEL_COLUMN,
    preprocess_dataframe,
)


class TestPipelineBuilding:

    def test_builds_without_error(self):
        pipeline = build_pipeline()
        assert isinstance(pipeline, Pipeline)

    def test_has_expected_steps(self):
        pipeline = build_pipeline()
        step_names = [name for name, _ in pipeline.steps]
        assert "preprocessor" in step_names
        assert "classifier" in step_names


class TestTraining:

    def test_trains_successfully(self, small_valid_df):
        cleaned, err = preprocess_dataframe(small_valid_df, require_label=True)
        assert err is None
        pipeline, X_test, y_test, err = train_model(cleaned)
        assert err is None
        assert pipeline is not None
        assert X_test is not None
        assert y_test is not None

    def test_train_on_synthetic(self, synthetic_df):
        cleaned, err = preprocess_dataframe(synthetic_df, require_label=True)
        assert err is None
        pipeline, X_test, y_test, err = train_model(cleaned)
        assert err is None
        assert pipeline is not None

    def test_test_set_is_held_out(self, small_valid_df):
        """Test-set size should be approximately 25% of total."""
        cleaned, _ = preprocess_dataframe(small_valid_df, require_label=True)
        pipeline, X_test, y_test, _ = train_model(cleaned)
        expected_test_n = int(len(cleaned) * 0.25)
        # Allow ±2 rows due to stratification rounding
        assert abs(len(X_test) - expected_test_n) <= 2

    def test_train_fails_without_label(self, feature_only_df):
        pipeline, X_test, y_test, err = train_model(feature_only_df)
        assert err is not None
        assert pipeline is None

    def test_reproducible_predictions(self, synthetic_df):
        """Same data + same seed → same predictions."""
        cleaned, _ = preprocess_dataframe(synthetic_df, require_label=True)
        pipeline1, X_test1, y_test1, _ = train_model(cleaned)
        pipeline2, X_test2, y_test2, _ = train_model(cleaned)
        pred1, _, _ = predict(pipeline1, X_test1)
        pred2, _, _ = predict(pipeline2, X_test2)
        assert np.array_equal(pred1, pred2)


class TestPrediction:

    @pytest.fixture(scope="class")
    @classmethod
    def trained_pipeline(cls, synthetic_df):
        from src.data_processing import preprocess_dataframe
        cleaned, _ = preprocess_dataframe(synthetic_df, require_label=True)
        pipeline, X_test, y_test, _ = train_model(cleaned)
        return pipeline, X_test, y_test

    def test_predict_returns_array(self, trained_pipeline):
        pipeline, X_test, y_test = trained_pipeline
        y_pred, y_proba, err = predict(pipeline, X_test)
        assert err is None
        assert y_pred is not None
        assert len(y_pred) == len(X_test)

    def test_predict_proba_shape(self, trained_pipeline):
        pipeline, X_test, y_test = trained_pipeline
        y_pred, y_proba, err = predict(pipeline, X_test)
        assert y_proba is not None
        assert y_proba.shape[0] == len(X_test)
        assert y_proba.shape[1] >= 2

    def test_predictions_are_known_labels(self, trained_pipeline, synthetic_df):
        pipeline, X_test, y_test = trained_pipeline
        known_labels = set(synthetic_df[LABEL_COLUMN].unique())
        y_pred, _, _ = predict(pipeline, X_test)
        predicted_labels = set(y_pred)
        assert predicted_labels.issubset(known_labels)

    def test_proba_sums_to_one(self, trained_pipeline):
        pipeline, X_test, y_test = trained_pipeline
        _, y_proba, _ = predict(pipeline, X_test)
        row_sums = y_proba.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-5)


class TestModelInfo:

    def test_get_model_info(self, synthetic_df):
        from src.data_processing import preprocess_dataframe
        cleaned, _ = preprocess_dataframe(synthetic_df, require_label=True)
        pipeline, _, _, _ = train_model(cleaned)
        info = get_model_info(pipeline)
        assert "model_type" in info
        assert "n_estimators" in info

    def test_get_feature_importance(self, synthetic_df):
        from src.data_processing import preprocess_dataframe
        cleaned, _ = preprocess_dataframe(synthetic_df, require_label=True)
        pipeline, _, _, _ = train_model(cleaned)
        fi = get_feature_importance(pipeline)
        assert fi is not None
        assert "feature" in fi.columns
        assert "importance" in fi.columns
        assert len(fi) > 0
        # Importances should sum to approximately 1
        assert abs(fi["importance"].sum() - 1.0) < 0.01

