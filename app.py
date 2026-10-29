"""
app.py — Cyber Guardian AI
--------------------------
Main Streamlit application entry point.

Run with:
    streamlit run app.py

This application demonstrates machine-learning-based network-traffic
classification for educational purposes. It is NOT a production
intrusion prevention system.

Project: AI-Powered Cyber Attacks and Defence
Group:   Cyber Nexus
"""

from __future__ import annotations

import sys
import os

# Ensure the project root is on sys.path so `src` is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import io
import logging
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.data_processing import (
    LABEL_COLUMN,
    REQUIRED_FEATURE_COLUMNS,
    generate_synthetic_dataset,
    get_class_distribution,
    get_dataset_summary,
    load_csv_bytes,
    preprocess_dataframe,
)
from src.detection import (
    EvaluationResult,
    TrainingResult,
    get_threat_summary,
    run_evaluation,
    run_prediction,
    run_training,
)
from src.evaluation import format_fp_fn_table, interpret_metrics
from src.visualizations import (
    plot_class_distribution_bar,
    plot_class_distribution_pie,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_metrics_radar,
    plot_per_class_metrics,
    plot_prediction_distribution,
    plot_risk_gauge,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Cyber Guardian AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/harshalharde45/Seminaraiphishing",
        "Report a bug": "https://github.com/harshalharde45/Seminaraiphishing/issues",
        "About": "Cyber Guardian AI — Educational ML Cybersecurity Prototype. Project Group: Cyber Nexus.",
    },
)

# ---------------------------------------------------------------------------
# Custom CSS — professional dark cybersecurity theme
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* === Global === */
html, body, [class*="css"] {
    font-family: 'Segoe UI', sans-serif;
}

/* === Sidebar === */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a0e1a 0%, #0f1629 100%);
    border-right: 1px solid rgba(0,212,255,0.15);
}
[data-testid="stSidebar"] .stMarkdown h2 {
    color: #00d4ff;
    font-size: 1.1rem;
    margin-top: 1rem;
}

/* === Metric cards === */
[data-testid="stMetric"] {
    background: #141c2e;
    border: 1px solid rgba(0,212,255,0.2);
    border-radius: 8px;
    padding: 12px 16px;
}
[data-testid="stMetricValue"] { color: #00d4ff !important; }
[data-testid="stMetricLabel"] { color: #a0aec0 !important; }
[data-testid="stMetricDelta"] { font-size: 0.8rem; }

/* === Section headers === */
.section-header {
    border-left: 4px solid #00d4ff;
    padding-left: 12px;
    margin: 1.5rem 0 1rem 0;
    color: #e8eaf6;
}

/* === Info/warning banners === */
.banner-info {
    background: rgba(0,212,255,0.08);
    border: 1px solid rgba(0,212,255,0.3);
    border-radius: 8px;
    padding: 12px 16px;
    margin: 0.5rem 0;
    color: #e8eaf6;
}
.banner-warn {
    background: rgba(255,179,71,0.08);
    border: 1px solid rgba(255,179,71,0.3);
    border-radius: 8px;
    padding: 12px 16px;
    margin: 0.5rem 0;
    color: #e8eaf6;
}
.banner-alert {
    background: rgba(255,75,110,0.08);
    border: 1px solid rgba(255,75,110,0.3);
    border-radius: 8px;
    padding: 12px 16px;
    margin: 0.5rem 0;
    color: #e8eaf6;
}
.banner-success {
    background: rgba(0,204,106,0.08);
    border: 1px solid rgba(0,204,106,0.3);
    border-radius: 8px;
    padding: 12px 16px;
    margin: 0.5rem 0;
    color: #e8eaf6;
}

/* === Data tags === */
.tag-benign  { background:#00d4ff22; color:#00d4ff; border:1px solid #00d4ff44; border-radius:4px; padding:2px 8px; font-size:0.8rem; }
.tag-attack  { background:#ff4b6e22; color:#ff4b6e; border:1px solid #ff4b6e44; border-radius:4px; padding:2px 8px; font-size:0.8rem; }
.tag-warning { background:#ffb34722; color:#ffb347; border:1px solid #ffb34744; border-radius:4px; padding:2px 8px; font-size:0.8rem; }

/* === Hero banner === */
.hero {
    background: linear-gradient(135deg, #0a0e1a 0%, #0f2040 50%, #0a1428 100%);
    border: 1px solid rgba(0,212,255,0.25);
    border-radius: 12px;
    padding: 28px 32px;
    margin-bottom: 1.5rem;
    text-align: center;
}
.hero h1 { color: #00d4ff; font-size: 2rem; margin: 0 0 4px 0; }
.hero p  { color: #a0aec0; margin: 4px 0; }

/* === Table styling === */
.dataframe { font-size: 0.85rem; }

/* === Button === */
.stButton > button {
    background: linear-gradient(135deg, #00b4cc, #00d4ff);
    color: #0a0e1a;
    border: none;
    border-radius: 6px;
    font-weight: 600;
    padding: 0.5rem 1.5rem;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #00d4ff, #00b4cc);
    color: #0a0e1a;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults = {
        "raw_df": None,
        "processed_df": None,
        "data_source": None,           # "synthetic" | "uploaded"
        "training_result": None,       # TrainingResult
        "eval_result": None,           # EvaluationResult
        "pred_result": None,           # PredictionResult
        "active_page": "🏠 Overview",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


_init_state()


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------

PAGES = [
    "🏠 Overview",
    "🔍 Data Explorer",
    "🧪 AI Detection Lab",
    "⚠️ Threat Analysis",
    "💡 Security Insights",
    "📊 Model Evaluation",
    "📚 Threat Intelligence",
    "ℹ️ About",
]

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 16px 0 8px 0;">
        <div style="font-size:2.5rem;">🛡️</div>
        <div style="color:#00d4ff; font-size:1.1rem; font-weight:700;">Cyber Guardian AI</div>
        <div style="color:#607080; font-size:0.75rem;">AI-Powered Cybersecurity</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    selected = st.radio(
        "Navigate",
        PAGES,
        index=PAGES.index(st.session_state.active_page),
        label_visibility="collapsed",
    )
    st.session_state.active_page = selected

    st.divider()

    # Status panel
    st.markdown("## Status")
    if st.session_state.raw_df is not None:
        src = st.session_state.data_source or "unknown"
        n = len(st.session_state.raw_df)
        st.success(f"✅ Data loaded ({n:,} records, {src})")
    else:
        st.info("📂 No data loaded")

    if st.session_state.training_result and st.session_state.training_result.success:
        st.success("✅ Model trained")
    else:
        st.warning("⏳ Model not trained")

    if st.session_state.eval_result and st.session_state.eval_result.success:
        acc = st.session_state.eval_result.metrics.get("accuracy", 0)
        st.success(f"✅ Evaluated ({acc*100:.1f}% acc)")
    else:
        st.info("📊 Not evaluated")

    st.divider()
    st.markdown(
        "<div style='color:#708090; font-size:0.72rem; line-height:1.4;'>"
        "<strong>Harshal Harde</strong> (PRN: 2125UCSM1001)<br>"
        "Cyber Nexus | Sanjivani University<br>"
        "Guide: Madhu B. Sir<br>"
        "<span style='color:#e67e22;'>⚠️ Synthetic Demo Data</span></div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Page rendering functions
# ---------------------------------------------------------------------------

def page_overview() -> None:
    st.markdown("""
    <div class="hero">
        <h1>🛡️ Cyber Guardian AI</h1>
        <p style="font-size:1.1rem; color:#e8eaf6;">AI-Powered Cyber Attacks and Defence</p>
        <p style="color:#607080;">Challenges, Detection, and Future Solutions &nbsp;|&nbsp; Cyber Nexus</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="banner-warn">
    ⚠️ <strong>Educational Prototype Notice:</strong> This application uses
    <strong>synthetically generated</strong> network-traffic data for demonstration.
    Results do NOT represent real-world intrusion detection performance.
    This is not a production security tool. Human analyst review is always required.
    </div>
    """, unsafe_allow_html=True)

    # === Quick-start ===
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown('<h3 class="section-header">Quick Start</h3>', unsafe_allow_html=True)
        qcol1, qcol2 = st.columns(2)
        with qcol1:
            if st.button("🔬 Load Demo Dataset", use_container_width=True):
                with st.spinner("Generating synthetic dataset…"):
                    df = generate_synthetic_dataset(n_benign=600, n_attack=400)
                    st.session_state.raw_df = df
                    st.session_state.data_source = "synthetic"
                    st.session_state.training_result = None
                    st.session_state.eval_result = None
                    st.session_state.pred_result = None
                st.success(f"✅ Synthetic dataset loaded — {len(df):,} records.")
                st.rerun()
        with qcol2:
            uploaded = st.file_uploader(
                "📤 Upload CSV",
                type=["csv"],
                label_visibility="collapsed",
                key="overview_uploader",
            )
            if uploaded:
                raw_bytes = uploaded.read()
                df, err = load_csv_bytes(raw_bytes)
                if err:
                    st.error(f"Upload error: {err}")
                else:
                    st.session_state.raw_df = df
                    st.session_state.data_source = "uploaded"
                    st.session_state.training_result = None
                    st.session_state.eval_result = None
                    st.session_state.pred_result = None
                    st.success(f"✅ Uploaded — {len(df):,} records.")
                    st.rerun()

    with col2:
        st.markdown('<h3 class="section-header">Dataset Summary</h3>', unsafe_allow_html=True)
        if st.session_state.raw_df is not None:
            df = st.session_state.raw_df
            summary = get_dataset_summary(df)
            st.metric("Total Records", f"{summary['total_records']:,}")
            st.metric("Columns", summary["total_columns"])
            if summary["has_label"]:
                st.metric("Classes", summary.get("num_classes", "—"))
            st.metric("Missing Values", summary["missing_values"])
        else:
            st.info("Load a dataset to see a summary.")

    # === Metrics row ===
    if st.session_state.eval_result and st.session_state.eval_result.success:
        st.markdown('<h3 class="section-header">Model Performance (Held-Out Test Set)</h3>', unsafe_allow_html=True)
        m = st.session_state.eval_result.metrics
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Accuracy", f"{m['accuracy']*100:.2f}%")
        c2.metric("Precision (Macro)", f"{m['precision_macro']:.4f}")
        c3.metric("Recall (Macro)", f"{m['recall_macro']:.4f}")
        c4.metric("F1 (Macro)", f"{m['f1_macro']:.4f}")
        c5.metric("Test Samples", m['test_samples'])

    # === Class distribution ===
    if st.session_state.raw_df is not None:
        df = st.session_state.raw_df
        if LABEL_COLUMN in df.columns:
            st.markdown('<h3 class="section-header">Class Distribution</h3>', unsafe_allow_html=True)
            dist = get_class_distribution(df)
            ccol1, ccol2 = st.columns(2)
            with ccol1:
                st.plotly_chart(plot_class_distribution_bar(dist), use_container_width=True)
            with ccol2:
                st.plotly_chart(plot_class_distribution_pie(dist), use_container_width=True)

    # === Pipeline diagram ===
    st.markdown('<h3 class="section-header">Detection Pipeline</h3>', unsafe_allow_html=True)
    st.markdown("""
    <div class="banner-info" style="font-family:monospace; line-height:2; font-size:0.85rem;">
    📥 Data Input
    &nbsp;→&nbsp; 🔍 Validation &amp; Preprocessing
    &nbsp;→&nbsp; 🔢 Feature Engineering
    &nbsp;→&nbsp; 🤖 ML Classification (Random Forest)
    &nbsp;→&nbsp; 📊 Risk Assessment
    &nbsp;→&nbsp; 👁️ Analyst Review
    </div>
    """, unsafe_allow_html=True)


def page_data_explorer() -> None:
    st.markdown('<h2 class="section-header">🔍 Data Explorer</h2>', unsafe_allow_html=True)

    # Upload or use existing
    uploaded = st.file_uploader("Upload a compatible CSV file", type=["csv"])
    if uploaded:
        raw_bytes = uploaded.read()
        df, err = load_csv_bytes(raw_bytes)
        if err:
            st.error(f"Could not read file: {err}")
            return
        st.session_state.raw_df = df
        st.session_state.data_source = "uploaded"
        st.session_state.training_result = None
        st.session_state.eval_result = None
        st.session_state.pred_result = None
        st.success(f"Loaded {len(df):,} rows and {len(df.columns)} columns.")

    if st.session_state.raw_df is None:
        st.info("No data loaded. Use 'Load Demo Dataset' on the Overview page, or upload a CSV here.")
        _show_expected_format()
        return

    df = st.session_state.raw_df
    summary = get_dataset_summary(df)

    # Dataset info cards
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", f"{summary['total_records']:,}")
    c2.metric("Columns", summary["total_columns"])
    c3.metric("Missing Values", summary["missing_values"])
    c4.metric("Duplicate Rows", summary["duplicate_rows"])

    st.divider()
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Preview", "📈 Class Distribution", "🔎 Column Info", "✅ Validation"])

    with tab1:
        n_show = st.slider("Rows to display", 5, min(100, len(df)), 20)
        st.dataframe(df.head(n_show), use_container_width=True)

    with tab2:
        if LABEL_COLUMN in df.columns:
            dist = get_class_distribution(df)
            st.dataframe(dist, use_container_width=True)
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(plot_class_distribution_bar(dist), use_container_width=True)
            with col2:
                st.plotly_chart(plot_class_distribution_pie(dist), use_container_width=True)
        else:
            st.warning(f"Column '{LABEL_COLUMN}' not found. Class distribution unavailable.")

    with tab3:
        col_info = pd.DataFrame({
            "Column": df.columns,
            "Dtype": [str(t) for t in df.dtypes.values],
            "Non-Null Count": df.notnull().sum().values,
            "Null Count": df.isnull().sum().values,
            "Unique Values": df.nunique().values,
            "Sample": [str(df[c].iloc[0]) if len(df) > 0 else "" for c in df.columns],
        })
        st.dataframe(col_info, use_container_width=True)

        missing_feat = [c for c in REQUIRED_FEATURE_COLUMNS if c not in df.columns]
        if missing_feat:
            st.warning(f"Missing expected feature columns: {missing_feat}")
        else:
            st.success("✅ All expected feature columns are present.")

    with tab4:
        cleaned_df, err = preprocess_dataframe(df, require_label=(LABEL_COLUMN in df.columns))
        if err:
            st.error(f"Validation failed: {err}")
        else:
            st.success(f"✅ Validation passed. {len(cleaned_df):,} records ready for analysis.")
            if summary["missing_values"] > 0:
                st.info(f"ℹ️ {summary['missing_values']} missing values were filled with column median/mode.")
            if summary["duplicate_rows"] > 0:
                st.warning(f"⚠️ {summary['duplicate_rows']} duplicate rows detected.")


def _show_expected_format() -> None:
    st.markdown("### Expected CSV Format")
    sample_cols = REQUIRED_FEATURE_COLUMNS + [LABEL_COLUMN]
    sample_row = {
        "duration": 10.5, "protocol_type": "tcp", "src_bytes": 1024,
        "dst_bytes": 2048, "land": 0, "wrong_fragment": 0, "urgent": 0,
        "hot": 1, "num_failed_logins": 0, "logged_in": 1,
        "num_compromised": 0, "count": 10, "srv_count": 10,
        "same_srv_rate": 0.9, "diff_srv_rate": 0.1,
        "dst_host_count": 100, "dst_host_srv_count": 100,
        "dst_host_same_srv_rate": 0.9, "dst_host_diff_srv_rate": 0.1,
        "dst_host_same_src_port_rate": 0.2, "label": "BENIGN",
    }
    sample_df = pd.DataFrame([sample_row])
    st.dataframe(sample_df, use_container_width=True)
    st.caption("The `label` column is required for training but optional for prediction.")


def page_detection_lab() -> None:
    st.markdown('<h2 class="section-header">🧪 AI Detection Lab</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class="banner-info">
    This lab trains a <strong>Random Forest classifier</strong> on labelled traffic data,
    then evaluates it on a held-out test set (25% of the data).
    The model is used only for classification — it does not make decisions autonomously.
    </div>
    """, unsafe_allow_html=True)

    # Load data section
    st.markdown("### Step 1 — Load Data")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔬 Generate Synthetic Dataset", use_container_width=True):
            with st.spinner("Generating…"):
                df = generate_synthetic_dataset(n_benign=600, n_attack=400)
                st.session_state.raw_df = df
                st.session_state.data_source = "synthetic"
                st.session_state.training_result = None
                st.session_state.eval_result = None
                st.session_state.pred_result = None
            st.success(f"✅ {len(df):,} synthetic records generated.")

    with col2:
        uploaded = st.file_uploader("Or upload your CSV", type=["csv"])
        if uploaded:
            raw_bytes = uploaded.read()
            df, err = load_csv_bytes(raw_bytes)
            if err:
                st.error(err)
            else:
                st.session_state.raw_df = df
                st.session_state.data_source = "uploaded"
                st.session_state.training_result = None
                st.session_state.eval_result = None
                st.session_state.pred_result = None
                st.success(f"✅ Uploaded {len(df):,} records.")

    if st.session_state.raw_df is None:
        st.info("Please load data first.")
        return

    df = st.session_state.raw_df
    st.caption(
        f"Current dataset: **{len(df):,} records** "
        f"({st.session_state.data_source}) | "
        f"Label column: {'✅ present' if LABEL_COLUMN in df.columns else '❌ missing'}"
    )

    # Train section
    st.markdown("### Step 2 — Train the Model")

    if LABEL_COLUMN not in df.columns:
        st.error(
            f"The dataset does not have a '{LABEL_COLUMN}' column. "
            "Training requires labelled data. Upload a file with a label column "
            "or use the synthetic dataset."
        )
        return

    col_train, col_info = st.columns([2, 3])
    with col_train:
        train_btn = st.button("🚀 Train Classifier", use_container_width=True, type="primary")

    if train_btn:
        with st.spinner("Preprocessing and training… this may take a moment."):
            result = run_training(df)
            st.session_state.training_result = result
            if result.success:
                eval_result = run_evaluation(result.pipeline, result.X_test, result.y_test)
                st.session_state.eval_result = eval_result

                # Auto-run prediction on test set for threat analysis
                pred_df = result.X_test.copy()
                pred_df[LABEL_COLUMN] = result.y_test.values
                pred_result = run_prediction(result.pipeline, pred_df, has_labels=True)
                st.session_state.pred_result = pred_result

    tr = st.session_state.training_result
    if tr is None:
        with col_info:
            st.info("Click 'Train Classifier' to begin.")
        return

    if not tr.success:
        st.error(f"Training failed: {tr.error}")
        return

    with col_info:
        st.success(f"✅ Model trained successfully.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Train Samples", f"{tr.train_samples:,}")
        c2.metric("Test Samples", f"{tr.test_samples:,}")
        c3.metric("Classes", len(tr.classes))

    # Model info
    st.markdown("### Model Configuration")
    info = tr.model_info
    mi_cols = st.columns(4)
    mi_cols[0].info(f"**Type:** {info.get('model_type','RF')}")
    mi_cols[1].info(f"**Trees:** {info.get('n_estimators',150)}")
    mi_cols[2].info(f"**Max Depth:** {info.get('max_depth','∞')}")
    mi_cols[3].info(f"**Test Split:** {info.get('test_size','25%')}")

    # Quick metrics
    ev = st.session_state.eval_result
    if ev and ev.success:
        m = ev.metrics
        st.markdown("### Quick Evaluation (Held-Out Test Set)")
        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Accuracy", f"{m['accuracy']*100:.2f}%")
        mc2.metric("Precision (Macro)", f"{m['precision_macro']:.4f}")
        mc3.metric("Recall (Macro)", f"{m['recall_macro']:.4f}")
        mc4.metric("F1 (Macro)", f"{m['f1_macro']:.4f}")

        st.info("Go to **📊 Model Evaluation** for the full confusion matrix and per-class metrics.")

    # Predict on new data
    st.markdown("### Step 3 — Predict on New Records")
    st.caption(
        "Upload a CSV with feature columns (label column optional). "
        "The model will classify each record."
    )
    pred_upload = st.file_uploader("Upload records for classification", type=["csv"], key="pred_uploader")
    if pred_upload and tr.success:
        raw_bytes = pred_upload.read()
        new_df, err = load_csv_bytes(raw_bytes)
        if err:
            st.error(err)
        else:
            with st.spinner("Classifying…"):
                pred_result = run_prediction(tr.pipeline, new_df)
            if not pred_result.success:
                st.error(f"Prediction error: {pred_result.error}")
            else:
                st.success(f"✅ Classified {len(pred_result.result_df):,} records.")
                st.session_state.pred_result = pred_result
                st.dataframe(
                    pred_result.result_df[["predicted_label", "risk_level", "confidence_score"]].head(50),
                    use_container_width=True,
                )


def page_threat_analysis() -> None:
    st.markdown('<h2 class="section-header">⚠️ Threat Analysis</h2>', unsafe_allow_html=True)

    tr = st.session_state.training_result
    ev = st.session_state.eval_result
    pr = st.session_state.pred_result

    if tr is None or not tr.success:
        st.info("Train the model first in the **🧪 AI Detection Lab**.")
        return

    # Prediction results
    if pr and pr.success:
        summary = get_threat_summary(pr.result_df)
        st.markdown("### Traffic Classification Summary")

        gcol1, gcol2 = st.columns([1, 2])
        with gcol1:
            st.plotly_chart(
                plot_risk_gauge(summary.get("attack_pct", 0)),
                use_container_width=True,
            )
        with gcol2:
            c1, c2, c3 = st.columns(3)
            c1.metric("Total Records", f"{summary.get('total',0):,}")
            c2.metric("Benign Traffic", f"{summary.get('benign_count',0):,}",
                      delta=f"{summary.get('benign_pct',0):.1f}%")
            c3.metric("Flagged Traffic", f"{summary.get('attack_count',0):,}",
                      delta=f"{summary.get('attack_pct',0):.1f}%")

        st.markdown("""
        <div class="banner-warn">
        ⚠️ <strong>Important:</strong> Flagged records are <em>model predictions</em>,
        not confirmed attacks. False positives and false negatives are expected.
        Human analyst review is required before any action is taken.
        </div>
        """, unsafe_allow_html=True)

        st.plotly_chart(
            plot_prediction_distribution(pr.result_df),
            use_container_width=True,
        )

        st.markdown("### Prediction Detail (First 100 Records)")
        display_cols = ["predicted_label", "risk_level", "confidence_score"]
        available_cols = [c for c in display_cols if c in pr.result_df.columns]
        st.dataframe(pr.result_df[available_cols].head(100), use_container_width=True)

    # Confusion matrix
    if ev and ev.success:
        st.divider()
        st.markdown("### Confusion Matrix")
        m = ev.metrics
        st.plotly_chart(
            plot_confusion_matrix(m["confusion_matrix"], m["classes"]),
            use_container_width=True,
        )

        st.markdown("""
        **Reading the confusion matrix:**
        - **Diagonal cells** (top-left to bottom-right): Correct predictions.
        - **Off-diagonal cells**: Misclassifications.
        - A high number in an off-diagonal cell means the model is confusing those two classes.
        - Each cell shows the count and the percentage of the true-class total.
        """)

        # FP/FN table
        fp_fn_df = format_fp_fn_table(m)
        if not fp_fn_df.empty:
            st.markdown("### False Positives and False Negatives per Class")
            st.dataframe(fp_fn_df, use_container_width=True)
            st.caption(
                "FP = benign/other traffic wrongly classified as this class | "
                "FN = this class wrongly classified as benign/other | "
                "TP = correctly identified | TN = correctly excluded"
            )


def page_security_insights() -> None:
    st.markdown('<h2 class="section-header">💡 Security Insights</h2>', unsafe_allow_html=True)

    ev = st.session_state.eval_result
    pr = st.session_state.pred_result
    df = st.session_state.raw_df

    if ev is None or not ev.success:
        st.info("Complete model training and evaluation to generate insights.")
        return

    m = ev.metrics

    # Computed observations
    st.markdown("### Computed Observations from Model Output")
    observations = interpret_metrics(m)
    for obs in observations:
        if "⚠️" in obs or "Warning" in obs.lower():
            st.warning(obs)
        elif "false negative" in obs.lower() or "false positive" in obs.lower():
            st.error(obs)
        else:
            st.info(obs)

    # Feature importance
    if ev.feature_importance is not None and not ev.feature_importance.empty:
        st.markdown("### Most Influential Features")
        st.plotly_chart(
            plot_feature_importance(ev.feature_importance),
            use_container_width=True,
        )
        st.caption(
            "Feature importance measures how much each feature reduces classification "
            "uncertainty in the Random Forest. Higher values → more discriminative features. "
            "This reflects patterns in the training data, not universal threat signals."
        )

    # Dataset-derived observations
    if df is not None and LABEL_COLUMN in df.columns:
        st.markdown("### Dataset-Derived Observations")
        dist = get_class_distribution(df)
        benign_pct = dist.loc[dist["Label"] == "BENIGN", "Percentage"].values
        if len(benign_pct) > 0:
            bp = benign_pct[0]
            if bp > 70:
                st.info(
                    f"📊 {bp:.1f}% of records are labelled BENIGN. "
                    "Class imbalance of this degree can bias the model toward predicting the majority class. "
                    "The 'balanced' class_weight setting in the Random Forest partially compensates."
                )

    # General educational advice
    st.divider()
    st.markdown("### General Security Principles")
    st.markdown("""
    The following principles are well-established in the cybersecurity field.
    They are general best practices, not derived from this specific dataset.

    - **Layered defence (Defence in Depth):** No single control is sufficient.
    - **Least privilege:** Grant only the access required for a task.
    - **Anomaly + signature detection:** Combine rule-based and ML approaches.
    - **Human-in-the-loop:** Automated alerts require analyst confirmation.
    - **Continuous monitoring:** Attackers adapt; so must defences.
    - **Patch management:** Many breaches exploit known, unpatched vulnerabilities.
    """)


def page_model_evaluation() -> None:
    st.markdown('<h2 class="section-header">📊 Model Evaluation</h2>', unsafe_allow_html=True)

    ev = st.session_state.eval_result
    tr = st.session_state.training_result

    if ev is None or not ev.success:
        st.info("Train the model first in the **🧪 AI Detection Lab**.")
        return

    m = ev.metrics

    # Metric overview
    st.markdown("### Overall Metrics (Held-Out Test Set)")
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("Accuracy", f"{m['accuracy']*100:.2f}%")
    mc2.metric("Precision (Macro)", f"{m['precision_macro']:.4f}")
    mc3.metric("Recall (Macro)", f"{m['recall_macro']:.4f}")
    mc4.metric("F1 (Macro)", f"{m['f1_macro']:.4f}")
    mc5.metric("F1 (Weighted)", f"{m['f1_weighted']:.4f}")
    st.caption(f"Evaluated on {m['test_samples']} held-out test records (25% stratified split).")

    st.markdown("""
    <div class="banner-info">
    <strong>Metric definitions:</strong><br>
    • <em>Accuracy</em>: Proportion of all predictions that are correct.<br>
    • <em>Precision (macro)</em>: Average across classes of TP/(TP+FP). How reliable are positive predictions?<br>
    • <em>Recall (macro)</em>: Average across classes of TP/(TP+FN). What fraction of actual positives are caught?<br>
    • <em>F1-score</em>: Harmonic mean of Precision and Recall. Balances both concerns.<br>
    • <em>Macro avg</em>: Treats all classes equally regardless of size.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Confusion Matrix", "📋 Per-Class Metrics", "🌐 Radar Chart", "🔬 Feature Importance"
    ])

    with tab1:
        st.plotly_chart(
            plot_confusion_matrix(m["confusion_matrix"], m["classes"]),
            use_container_width=True,
        )

    with tab2:
        st.dataframe(ev.per_class_df, use_container_width=True)
        st.plotly_chart(
            plot_per_class_metrics(ev.per_class_df),
            use_container_width=True,
        )

    with tab3:
        st.plotly_chart(
            plot_metrics_radar(m),
            use_container_width=True,
        )

    with tab4:
        if ev.feature_importance is not None and not ev.feature_importance.empty:
            st.plotly_chart(
                plot_feature_importance(ev.feature_importance),
                use_container_width=True,
            )
            st.dataframe(ev.feature_importance.head(20), use_container_width=True)
        else:
            st.info("Feature importance not available.")

    # Limitations
    st.divider()
    st.markdown("### Known Limitations")
    st.markdown("""
    1. **Synthetic data**: The dataset is programmatically generated, not real network traffic.
       Performance on synthetic data does not predict real-world effectiveness.
    2. **Distribution shift**: A model trained on synthetic data may perform poorly on traffic
       from different networks, time periods, or attack variants.
    3. **Class imbalance**: Even with balanced class weights, minority classes may have
       higher misclassification rates.
    4. **No temporal modelling**: Network flows are treated independently; temporal patterns
       (e.g., coordinated attack waves) are not captured.
    5. **Feature scope**: Only 20 features are used. Real-world detection may require hundreds.
    6. **Calibration**: Model probabilities are not calibrated. Do not interpret them as
       a verified probability that an attack is occurring.
    7. **Adversarial robustness**: A trained attacker could craft traffic to evade detection.
    """)

    for w in m.get("warnings", []):
        st.warning(w)


def page_threat_intelligence() -> None:
    st.markdown('<h2 class="section-header">📚 Educational Threat Intelligence</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class="banner-info">
    The topics below are covered as <strong>educational material</strong>.
    Cyber Guardian AI does not implement detectors for AI-generated phishing,
    deepfakes, malware, or prompt injection. These topics illustrate the
    broader landscape of AI threats that motivate AI-assisted defence research.
    </div>
    """, unsafe_allow_html=True)

    threats = {
        "🎣 AI-Generated Phishing": {
            "icon": "🎣",
            "description": (
                "Large language models can generate grammatically correct, contextually "
                "plausible phishing emails at scale, defeating traditional grammar-error "
                "heuristics. Spear-phishing attacks personalised with publicly available "
                "profile data are increasingly difficult to distinguish from legitimate "
                "communication."
            ),
            "defence": [
                "Email authentication (SPF, DKIM, DMARC)",
                "ML-based content anomaly detection",
                "Link analysis and URL reputation services",
                "User education and simulated phishing exercises",
            ],
        },
        "🎭 Deepfake Fraud": {
            "icon": "🎭",
            "description": (
                "AI-generated synthetic audio and video can impersonate executives, "
                "colleagues, or officials in real-time video calls. Documented cases "
                "include fraudulent wire transfers authorised after deepfake video "
                "conference calls."
            ),
            "defence": [
                "Multi-factor authentication for financial authorisations",
                "Out-of-band verbal confirmation for large transfers",
                "Deepfake detection tools (liveness detection, compression artefact analysis)",
                "Process controls: no single-person authorisation for large transactions",
            ],
        },
        "🦠 AI-Assisted Malware": {
            "icon": "🦠",
            "description": (
                "AI tools assist adversaries in generating polymorphic malware variants "
                "that evade signature-based detection. AI can also suggest exploit "
                "payloads and generate evasion code faster than manual development."
            ),
            "defence": [
                "Behaviour-based endpoint detection (EDR)",
                "Application sandboxing and dynamic analysis",
                "Strict application allowlisting",
                "Prompt injection controls for AI coding assistants",
            ],
        },
        "💉 Prompt Injection": {
            "icon": "💉",
            "description": (
                "Attackers embed hidden instructions in external content (websites, "
                "documents, emails) that an AI assistant reads. When processed, the "
                "injected instructions override the legitimate user's intent, causing "
                "the AI to exfiltrate data, perform unauthorised actions, or produce "
                "harmful output."
            ),
            "defence": [
                "Input sanitisation and output validation in AI pipelines",
                "Principle of least privilege for AI agent permissions",
                "Separation between instruction channels and data channels",
                "Human review of AI-initiated actions",
            ],
        },
        "🤖 Automated Social Engineering": {
            "icon": "🤖",
            "description": (
                "AI enables scalable, personalised social engineering. Chatbots can "
                "sustain multi-turn conversations to build trust before requesting "
                "credentials or sensitive information. Vishing (voice phishing) bots "
                "use cloned voices to impersonate known contacts."
            ),
            "defence": [
                "Identity verification protocols before sharing sensitive information",
                "Security awareness training with AI-generated scenario examples",
                "Multi-channel verification (do not trust a single communication channel)",
            ],
        },
    }

    for threat_name, data in threats.items():
        with st.expander(threat_name, expanded=False):
            st.markdown(f"**Overview:** {data['description']}")
            st.markdown("**Defensive Countermeasures:**")
            for d in data["defence"]:
                st.markdown(f"- {d}")

    st.divider()
    st.markdown("### AI in Defensive Cybersecurity")
    st.markdown("""
    | Application | Description |
    |---|---|
    | Network intrusion detection | Classify traffic flows as benign or malicious (implemented in this prototype) |
    | Anomaly detection | Identify statistical deviations from normal behaviour baselines |
    | Malware classification | Static and dynamic analysis of executable files |
    | User/Entity Behaviour Analytics (UEBA) | Detect compromised accounts by modelling normal behaviour |
    | Automated threat hunting | Assist analysts in querying large volumes of telemetry |
    | Vulnerability prioritisation | Rank vulnerabilities by exploitability and impact |
    | Incident response automation | Suggest containment actions; do NOT auto-execute without review |

    **Key principle:** AI tools *assist* human security analysts. They do not replace
    analyst judgement, organisational processes, or layered technical controls.
    """)


def page_about() -> None:
    st.markdown('<h2 class="section-header">ℹ️ About Cyber Guardian AI</h2>', unsafe_allow_html=True)

    st.markdown("""
    ## Project Overview

    **Cyber Guardian AI** is an educational prototype demonstrating how machine learning
    can assist with network-traffic threat classification.

    | Field | Value |
    |---|---|
    | **Student Name** | Harshal Harde |
    | **PRN** | 2125UCSM1001 |
    | **Project Title** | AI-Powered Cyber Attacks and Defence |
    | **Subtitle** | Challenges, Detection, and Future Solutions |
    | **Framework** | Cyber Guardian AI |
    | **Project Group** | Cyber Nexus |
    | **Guide** | Madhu B. Sir, Sanjivani University |
    | **Institutional Context** | Seminar Project, Sanjivani University |
    | **Domain** | Cybersecurity / AI in Cybersecurity |
    | **Repository** | [GitHub: harshalharde45/Seminaraiphishing](https://github.com/harshalharde45/Seminaraiphishing) |

    ## What Is Implemented

    ✅ Synthetic network-traffic dataset generation (clearly labelled as synthetic)  
    ✅ Data validation, preprocessing, and class-distribution analysis  
    ✅ Random Forest classification pipeline (scikit-learn)  
    ✅ Stratified 75/25 train/test split with fixed random seed  
    ✅ Actual (not hard-coded) evaluation metrics: accuracy, precision, recall, F1  
    ✅ Confusion matrix with per-class breakdown  
    ✅ False positive / false negative analysis  
    ✅ Feature importance ranking  
    ✅ Risk gauge and prediction confidence scores  
    ✅ CSV upload with schema validation  
    ✅ Educational threat intelligence coverage  

    ## What Is NOT Implemented

    ❌ Real-time network packet capture or live traffic monitoring  
    ❌ Deep learning or neural network classifiers  
    ❌ Deepfake detection  
    ❌ Phishing URL or email content analysis  
    ❌ Malware binary analysis  
    ❌ Prompt injection detection  
    ❌ Production-grade incident response or alerting  

    ## Architecture

    ```
    User Interface (Streamlit)
         │
         ├── Data Input: synthetic generator or CSV upload
         │
         ├── src/data_processing.py  ← validation, cleaning, feature extraction
         │
         ├── src/model.py            ← scikit-learn pipeline (ColumnTransformer + RF)
         │
         ├── src/detection.py        ← orchestration: train → evaluate → predict
         │
         ├── src/evaluation.py       ← metrics, confusion matrix, FP/FN analysis
         │
         └── src/visualizations.py  ← Plotly and Matplotlib charts
    ```

    ## Technology Stack

    | Component | Library | Version |
    |---|---|---|
    | Web UI | Streamlit | 1.40.0 |
    | ML Pipeline | scikit-learn | 1.5.2 |
    | Data Processing | pandas | 2.2.3 |
    | Numeric Computing | NumPy | 2.5.4 |
    | Interactive Charts | Plotly | 5.24.1 |
    | Static Charts | Matplotlib | 3.9.2 |
    | Testing | pytest | 8.3.3 |
    | Language | Python | 3.12.10 |

    ## Responsible Use Statement

    This application is an educational prototype for academic demonstration.

    - It is **not** a certified security product.
    - Its predictions are **not** confirmed security incidents.
    - Do **not** use it to justify blocking traffic or taking enforcement action.
    - Do **not** present synthetic-data accuracy as evidence of real-world effectiveness.
    - Always combine automated analysis with human expert review.
    - Ensure any uploaded data complies with applicable privacy and data-handling regulations.

    ## License

    MIT License — see [LICENSE](https://github.com/harshalharde45/Seminaraiphishing/blob/main/LICENSE) for details.
    """)


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

PAGE_MAP = {
    "🏠 Overview":             page_overview,
    "🔍 Data Explorer":        page_data_explorer,
    "🧪 AI Detection Lab":     page_detection_lab,
    "⚠️ Threat Analysis":      page_threat_analysis,
    "💡 Security Insights":    page_security_insights,
    "📊 Model Evaluation":     page_model_evaluation,
    "📚 Threat Intelligence":  page_threat_intelligence,
    "ℹ️ About":                page_about,
}


def main() -> None:
    page_fn = PAGE_MAP.get(st.session_state.active_page, page_overview)
    page_fn()


if __name__ == "__main__":
    main()
