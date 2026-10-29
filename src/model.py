"""
model.py
--------
Scikit-learn pipeline for training and predicting network-traffic classification.
Implements a Random Forest classifier with appropriate preprocessing.

Design decisions
----------------
- Categorical features (protocol_type) are one-hot encoded.
- Numeric features are standardised with StandardScaler.
- A ColumnTransformer + Pipeline ensures no data leakage between train and test.
- A fixed random_state guarantees reproducibility.
- The test set is used only for evaluation, never for hyperparameter tuning.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data_processing import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    get_features_and_labels,
)

logger = logging.getLogger(__name__)

RANDOM_STATE = 42
TEST_SIZE = 0.25  # 25 % held-out test set


# ---------------------------------------------------------------------------
# Pipeline factory
# ---------------------------------------------------------------------------

def build_pipeline() -> Pipeline:
    """
    Construct and return a scikit-learn Pipeline with preprocessing and
    a Random Forest classifier.

    The pipeline handles:
    - Numeric features: imputation with median (safety net) + StandardScaler.
    - Categorical features: most-frequent imputation + OneHotEncoder.
    """
    numeric_features = NUMERIC_FEATURES
    categorical_features = CATEGORICAL_FEATURES

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",           # drop any unlisted columns
        verbose_feature_names_out=False,
    )

    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",   # handles class imbalance
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", clf),
    ])

    return pipeline


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    df: pd.DataFrame,
) -> Tuple[Optional[Pipeline], Optional[pd.DataFrame], Optional[pd.Series], Optional[str]]:
    """
    Split the dataset and train the classification pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        A preprocessed DataFrame containing features and the label column.

    Returns
    -------
    (pipeline, X_test, y_test, None) on success.
    (None, None, None, error_message) on failure.
    """
    X, y, err = get_features_and_labels(df)
    if err:
        return None, None, None, err

    # Only keep feature columns that actually exist in X
    available_numeric = [c for c in NUMERIC_FEATURES if c in X.columns]
    available_categorical = [c for c in CATEGORICAL_FEATURES if c in X.columns]

    if not available_numeric and not available_categorical:
        return None, None, None, "No usable feature columns found in the dataset."

    # Stratified split to preserve class proportions
    try:
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
        train_idx, test_idx = next(splitter.split(X, y))
    except ValueError as exc:
        return None, None, None, f"Could not perform stratified split: {exc}"

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Build the pipeline with only available columns
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers = []
    if available_numeric:
        transformers.append(("num", numeric_transformer, available_numeric))
    if available_categorical:
        transformers.append(("cat", categorical_transformer, available_categorical))

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )

    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", clf),
    ])

    try:
        pipeline.fit(X_train, y_train)
    except Exception as exc:
        return None, None, None, f"Model training failed: {exc}"

    logger.info(
        "Model trained on %d samples, test set has %d samples.",
        len(X_train), len(X_test),
    )
    return pipeline, X_test, y_test, None


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict(
    pipeline: Pipeline,
    X: pd.DataFrame,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[str]]:
    """
    Run the pipeline on feature data and return predicted labels and
    class probabilities.

    Parameters
    ----------
    pipeline : Pipeline
        A fitted scikit-learn Pipeline.
    X : pd.DataFrame
        Feature DataFrame (must contain the expected columns).

    Returns
    -------
    (y_pred, y_proba, None) on success, or (None, None, error) on failure.
    y_proba shape: (n_samples, n_classes)
    """
    try:
        y_pred = pipeline.predict(X)
        y_proba = pipeline.predict_proba(X)
    except Exception as exc:
        return None, None, f"Prediction failed: {exc}"

    return y_pred, y_proba, None


def get_feature_importance(
    pipeline: Pipeline,
) -> Optional[pd.DataFrame]:
    """
    Extract feature importances from the Random Forest classifier.

    Returns a DataFrame with columns ['feature', 'importance'], sorted
    descending by importance, or None if extraction fails.
    """
    try:
        clf = pipeline.named_steps["classifier"]
        preprocessor = pipeline.named_steps["preprocessor"]
        feature_names = preprocessor.get_feature_names_out()
        importances = clf.feature_importances_
        df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return df
    except Exception as exc:
        logger.warning("Could not extract feature importances: %s", exc)
        return None


def get_model_info(pipeline: Pipeline) -> Dict[str, Any]:
    """Return a dictionary of model hyperparameters and metadata."""
    try:
        clf = pipeline.named_steps["classifier"]
        return {
            "model_type": "Random Forest Classifier",
            "n_estimators": clf.n_estimators,
            "max_depth": clf.max_depth if clf.max_depth else "None (unlimited)",
            "min_samples_split": clf.min_samples_split,
            "min_samples_leaf": clf.min_samples_leaf,
            "class_weight": str(clf.class_weight),
            "random_state": clf.random_state,
            "test_size": f"{int(TEST_SIZE * 100)}%",
        }
    except Exception:
        return {"model_type": "Random Forest Classifier"}

