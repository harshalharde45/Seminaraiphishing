"""
visualizations.py
-----------------
All chart-rendering functions for Cyber Guardian AI.

Uses Plotly for interactive charts and Matplotlib/Seaborn for
static matplotlib charts (e.g., confusion matrix heatmap).

All functions receive pre-computed data; they do not query the model.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette (cybersecurity theme)
# ---------------------------------------------------------------------------

COLOURS = {
    "cyan": "#00d4ff",
    "teal": "#00b4cc",
    "navy": "#0a0e1a",
    "dark_blue": "#0f1629",
    "panel": "#141c2e",
    "text": "#e8eaf6",
    "benign": "#00d4ff",
    "attack": "#ff4b6e",
    "warning": "#ffb347",
    "success": "#00cc6a",
    "grid": "rgba(255,255,255,0.05)",
}

ATTACK_COLOUR_MAP = {
    "BENIGN": COLOURS["benign"],
    "DoS": COLOURS["attack"],
    "DDoS": "#ff6b4b",
    "PortScan": "#ffb347",
    "BruteForce": "#c44dff",
    "Web Attack": "#ff4b8e",
    "Infiltration": "#e04bff",
    "Botnet": "#ff4b4b",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor=COLOURS["navy"],
    plot_bgcolor=COLOURS["panel"],
    font=dict(color=COLOURS["text"], family="sans-serif", size=12),
    margin=dict(l=40, r=40, t=50, b=40),
    xaxis=dict(gridcolor=COLOURS["grid"], zerolinecolor=COLOURS["grid"]),
    yaxis=dict(gridcolor=COLOURS["grid"], zerolinecolor=COLOURS["grid"]),
)


def _apply_layout(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(title=title, **PLOTLY_LAYOUT)
    return fig


# ---------------------------------------------------------------------------
# Class Distribution Charts
# ---------------------------------------------------------------------------

def plot_class_distribution_bar(class_dist_df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of label counts."""
    colours = [
        ATTACK_COLOUR_MAP.get(lbl, COLOURS["cyan"])
        for lbl in class_dist_df["Label"]
    ]
    fig = go.Figure(go.Bar(
        x=class_dist_df["Count"],
        y=class_dist_df["Label"],
        orientation="h",
        marker_color=colours,
        text=class_dist_df["Percentage"].apply(lambda p: f"{p:.1f}%"),
        textposition="outside",
    ))
    _apply_layout(fig, "Traffic Class Distribution")
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        xaxis_title="Record Count",
        yaxis_title="Class",
    )
    return fig


def plot_class_distribution_pie(class_dist_df: pd.DataFrame) -> go.Figure:
    """Donut chart of label proportions."""
    colours = [
        ATTACK_COLOUR_MAP.get(lbl, COLOURS["cyan"])
        for lbl in class_dist_df["Label"]
    ]
    fig = go.Figure(go.Pie(
        labels=class_dist_df["Label"],
        values=class_dist_df["Count"],
        hole=0.55,
        marker_colors=colours,
        textinfo="label+percent",
        textfont_size=11,
    ))
    _apply_layout(fig, "Class Proportions")
    return fig


# ---------------------------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List[str],
) -> go.Figure:
    """Interactive heatmap of the confusion matrix."""
    cm_norm = cm.astype(float)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    # Avoid division by zero
    cm_pct = np.where(row_sums > 0, cm_norm / row_sums * 100, 0)

    text_labels = [
        [f"{cm[i][j]}<br>({cm_pct[i][j]:.1f}%)" for j in range(len(classes))]
        for i in range(len(classes))
    ]

    fig = go.Figure(go.Heatmap(
        z=cm_pct,
        x=[f"Pred: {c}" for c in classes],
        y=[f"True: {c}" for c in classes],
        text=text_labels,
        texttemplate="%{text}",
        colorscale=[
            [0.0, COLOURS["panel"]],
            [0.5, COLOURS["teal"]],
            [1.0, COLOURS["cyan"]],
        ],
        showscale=True,
        colorbar=dict(title="% of True Class"),
    ))
    _apply_layout(fig, "Confusion Matrix (Held-Out Test Set)")
    fig.update_layout(
        xaxis=dict(title="Predicted Label", tickangle=-20),
        yaxis=dict(title="True Label", autorange="reversed"),
    )
    return fig


# ---------------------------------------------------------------------------
# Per-Class Metrics
# ---------------------------------------------------------------------------

def plot_per_class_metrics(per_class_df: pd.DataFrame) -> go.Figure:
    """Grouped bar chart of Precision, Recall, and F1-Score per class."""
    if per_class_df.empty:
        return go.Figure()

    fig = go.Figure()
    for metric, colour in [
        ("Precision", COLOURS["cyan"]),
        ("Recall", COLOURS["teal"]),
        ("F1-Score", "#c44dff"),
    ]:
        fig.add_trace(go.Bar(
            name=metric,
            x=per_class_df["Class"],
            y=per_class_df[metric],
            marker_color=colour,
        ))

    _apply_layout(fig, "Per-Class Precision, Recall, and F1-Score")
    fig.update_layout(
        barmode="group",
        yaxis=dict(range=[0, 1.1], title="Score"),
        xaxis_title="Traffic Class",
        legend=dict(orientation="h", y=1.1),
    )
    return fig


# ---------------------------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------------------------

def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_n: int = 15,
) -> go.Figure:
    """Horizontal bar chart of top feature importances."""
    df = importance_df.head(top_n).copy()
    df = df.sort_values("importance", ascending=True)

    fig = go.Figure(go.Bar(
        x=df["importance"],
        y=df["feature"],
        orientation="h",
        marker=dict(
            color=df["importance"],
            colorscale=[
                [0, COLOURS["dark_blue"]],
                [1, COLOURS["cyan"]],
            ],
            showscale=False,
        ),
    ))
    _apply_layout(fig, f"Top {top_n} Feature Importances (Random Forest)")
    fig.update_layout(
        xaxis_title="Mean Decrease in Impurity",
        yaxis_title="Feature",
    )
    return fig


# ---------------------------------------------------------------------------
# Prediction Results
# ---------------------------------------------------------------------------

def plot_prediction_distribution(result_df: pd.DataFrame) -> go.Figure:
    """Bar chart of prediction counts from a new analysis run."""
    if "predicted_label" not in result_df.columns:
        return go.Figure()

    counts = result_df["predicted_label"].value_counts().reset_index()
    counts.columns = ["Label", "Count"]

    colours = [
        ATTACK_COLOUR_MAP.get(lbl, COLOURS["cyan"])
        for lbl in counts["Label"]
    ]

    fig = go.Figure(go.Bar(
        x=counts["Label"],
        y=counts["Count"],
        marker_color=colours,
        text=counts["Count"],
        textposition="outside",
    ))
    _apply_layout(fig, "Predicted Traffic Classification")
    fig.update_layout(
        xaxis_title="Predicted Class",
        yaxis_title="Count",
    )
    return fig


def plot_risk_gauge(attack_pct: float) -> go.Figure:
    """Gauge chart showing the proportion of flagged (attack) traffic."""
    if attack_pct <= 20:
        bar_colour = COLOURS["success"]
        level = "LOW RISK"
    elif attack_pct <= 50:
        bar_colour = COLOURS["warning"]
        level = "MEDIUM RISK"
    else:
        bar_colour = COLOURS["attack"]
        level = "HIGH RISK"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=attack_pct,
        title={"text": f"Suspicious Traffic: {level}", "font": {"color": COLOURS["text"]}},
        number={"suffix": "%", "font": {"color": COLOURS["text"]}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": COLOURS["text"]},
            "bar": {"color": bar_colour},
            "steps": [
                {"range": [0, 20], "color": "rgba(0,204,106,0.15)"},
                {"range": [20, 50], "color": "rgba(255,179,71,0.15)"},
                {"range": [50, 100], "color": "rgba(255,75,110,0.15)"},
            ],
            "threshold": {
                "line": {"color": COLOURS["attack"], "width": 2},
                "thickness": 0.75,
                "value": 50,
            },
            "bgcolor": COLOURS["panel"],
            "bordercolor": COLOURS["grid"],
        },
    ))
    fig.update_layout(
        paper_bgcolor=COLOURS["navy"],
        font=dict(color=COLOURS["text"]),
        margin=dict(l=30, r=30, t=60, b=20),
        height=250,
    )
    return fig


# ---------------------------------------------------------------------------
# Metric Summary Radar
# ---------------------------------------------------------------------------

def plot_metrics_radar(metrics: Dict) -> go.Figure:
    """Radar chart showing key evaluation metrics."""
    categories = ["Accuracy", "Precision", "Recall", "F1 (Macro)", "F1 (Weighted)"]
    values = [
        metrics.get("accuracy", 0),
        metrics.get("precision_macro", 0),
        metrics.get("recall_macro", 0),
        metrics.get("f1_macro", 0),
        metrics.get("f1_weighted", 0),
    ]
    # Close the loop
    categories_closed = categories + [categories[0]]
    values_closed = values + [values[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(0,212,255,0.15)",
        line=dict(color=COLOURS["cyan"], width=2),
        name="Model Performance",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                gridcolor=COLOURS["grid"],
                color=COLOURS["text"],
            ),
            angularaxis=dict(color=COLOURS["text"]),
            bgcolor=COLOURS["panel"],
        ),
        paper_bgcolor=COLOURS["navy"],
        font=dict(color=COLOURS["text"]),
        margin=dict(l=60, r=60, t=60, b=60),
        title=dict(
            text="Model Performance Radar",
            font=dict(color=COLOURS["text"]),
        ),
    )
    return fig

