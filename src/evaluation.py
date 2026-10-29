"""
evaluation.py
-------------
Computes and formats evaluation metrics for the Cyber Guardian AI classifier.

All metrics are computed from ACTUAL model outputs.
Nothing is hard-coded.
"""

from __future__ import annotations

import warnings as py_warnings

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core Evaluation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[Optional[Dict], Optional[str]]:
    """
    Compute a comprehensive set of evaluation metrics.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels from the held-out test set.
    y_pred : array-like
        Predicted labels from the model.

    Returns
    -------
    (metrics_dict, None) on success, or (None, error_message) on failure.

    Notes
    -----
    - Uses macro averaging to treat each class equally regardless of size.
    - Uses zero_division=0 to avoid undefined metric warnings.
    - If only one unique class is present in y_true, metrics may be
      misleading; a warning is included in the returned dict.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if len(y_true) == 0:
        return None, "Cannot compute metrics: y_true is empty."

    if len(y_true) != len(y_pred):
        return None, (
            f"y_true has {len(y_true)} elements but y_pred has {len(y_pred)}."
        )

    classes = np.unique(np.concatenate([y_true, y_pred]))
    warnings = []

    if len(np.unique(y_true)) < 2:
        warnings.append(
            "Warning: only one class present in y_true. "
            "Metrics may be misleading."
        )

    try:
        accuracy = float(accuracy_score(y_true, y_pred))
        precision_macro = float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        )
        recall_macro = float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        )
        f1_macro = float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        )
        f1_weighted = float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        )
    except Exception as exc:
        return None, f"Metric computation error: {exc}"

    # Confusion matrix
    with py_warnings.catch_warnings():
        py_warnings.simplefilter("ignore", UserWarning)
        cm = confusion_matrix(y_true, y_pred, labels=list(classes))

    # Per-class metrics
    report = classification_report(
        y_true,
        y_pred,
        labels=list(classes),
        output_dict=True,
        zero_division=0,
    )

    # False positive / false negative summary (binary-style for each class)
    fp_fn = _compute_fp_fn(y_true, y_pred, classes)

    metrics = {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "classes": list(classes),
        "confusion_matrix": cm,
        "classification_report": report,
        "fp_fn_per_class": fp_fn,
        "test_samples": len(y_true),
        "warnings": warnings,
    }

    return metrics, None


def _compute_fp_fn(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: np.ndarray,
) -> List[Dict]:
    """Compute per-class FP, FN, TP, TN counts."""
    results = []
    for cls in classes:
        tp = int(np.sum((y_pred == cls) & (y_true == cls)))
        fp = int(np.sum((y_pred == cls) & (y_true != cls)))
        fn = int(np.sum((y_pred != cls) & (y_true == cls)))
        tn = int(np.sum((y_pred != cls) & (y_true != cls)))
        results.append({
            "class": cls,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
        })
    return results


def format_per_class_metrics(metrics: Dict) -> pd.DataFrame:
    """
    Convert the classification_report dict into a tidy DataFrame.

    Returns columns: Class, Precision, Recall, F1-Score, Support
    """
    report = metrics.get("classification_report", {})
    rows = []
    for label, values in report.items():
        if label in ("accuracy", "macro avg", "weighted avg"):
            continue
        if not isinstance(values, dict):
            continue
        rows.append({
            "Class": label,
            "Precision": round(values.get("precision", 0.0), 4),
            "Recall": round(values.get("recall", 0.0), 4),
            "F1-Score": round(values.get("f1-score", 0.0), 4),
            "Support": int(values.get("support", 0)),
        })
    return pd.DataFrame(rows)


def format_fp_fn_table(metrics: Dict) -> pd.DataFrame:
    """Return FP/FN table as a DataFrame."""
    fp_fn = metrics.get("fp_fn_per_class", [])
    if not fp_fn:
        return pd.DataFrame()
    return pd.DataFrame(fp_fn)


def interpret_metrics(metrics: Dict) -> List[str]:
    """
    Generate plain-language interpretations of evaluation results.

    All statements are derived from actual computed metrics.
    """
    observations = []
    acc = metrics.get("accuracy", 0)
    f1 = metrics.get("f1_macro", 0)
    prec = metrics.get("precision_macro", 0)
    rec = metrics.get("recall_macro", 0)

    observations.append(
        f"The model correctly classified {acc * 100:.1f}% of held-out test records."
    )
    observations.append(
        f"Macro-averaged F1-score: {f1:.3f} — balances precision and recall "
        f"equally across all {len(metrics.get('classes', []))} classes."
    )

    if prec > rec + 0.1:
        observations.append(
            "Precision is notably higher than recall: when the model predicts an "
            "attack, it is often correct, but it may miss some actual attacks "
            "(false negatives). In security contexts, false negatives can be "
            "dangerous because real threats go undetected."
        )
    elif rec > prec + 0.1:
        observations.append(
            "Recall is notably higher than precision: the model catches most "
            "attacks but generates excess false positives (benign traffic flagged "
            "as attacks). This increases analyst workload."
        )
    else:
        observations.append(
            "Precision and recall are relatively balanced across classes."
        )

    if acc >= 0.95:
        observations.append(
            "High accuracy on synthetic data does NOT imply the model will "
            "perform equally on real network traffic. Synthetic distributions "
            "may not capture real-world attack variation."
        )

    fp_fn = metrics.get("fp_fn_per_class", [])
    for row in fp_fn:
        if row["FN"] > 0 and row["class"] != "BENIGN":
            observations.append(
                f"Class '{row['class']}': {row['FN']} false negative(s) — "
                f"actual attacks classified as benign or another class."
            )
        if row["FP"] > 0 and row["class"] != "BENIGN":
            observations.append(
                f"Class '{row['class']}': {row['FP']} false positive(s) — "
                f"benign or other traffic mis-classified as this attack type."
            )

    for w in metrics.get("warnings", []):
        observations.append(f"⚠️ {w}")

    return observations

