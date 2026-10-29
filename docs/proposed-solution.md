# Proposed Solution — Cyber Guardian AI

## Architecture Overview
The proposed solution, **Cyber Guardian AI**, is designed around an end-to-end, human-in-the-loop threat identification framework. Rather than acting as a closed-box black box or claiming autonomous intervention, it functions as a security decision-support system.

## Modular Component Architecture

### 1. Ingestion & Validation Engine (`src/data_processing.py`)
- Ingests CSV network flow data or triggers the reproducible synthetic benchmark generator.
- Validates the presence of 20 essential flow features (flow duration, protocol types, payload volumes, login indicators, and connection frequency counters).
- Rejects malformed, incomplete, or corrupted records.
- Cleans and normalizes records by imputing missing values and replacing non-finite numbers without data leakage.

### 2. Scalable ML Classification Engine (`src/model.py`)
- Employs a composite scikit-learn `Pipeline` utilizing `ColumnTransformer`.
- Performs feature-level standard scaling on continuous metrics and one-hot encoding on categorical attributes (`protocol_type`).
- Leverages an ensemble of 150 randomized decision trees (`RandomForestClassifier`) equipped with cost-sensitive class balancing (`class_weight='balanced'`) to mitigate majority-class skew.
- Executes stratified splitting (75% training, 25% held-out testing) with a fixed seed (`random_state=42`).

### 3. Threat Detection & Orchestration (`src/detection.py`)
- Orchestrates training, batch prediction, and evaluation workflows.
- Translates raw classification probabilities into normalized confidence scores.
- Maps multiclass threat predictions into operational risk categories:
  - `BENIGN` $\rightarrow$ **LOW RISK**
  - Attack types (`DoS`, `DDoS`, `PortScan`, `BruteForce`) $\rightarrow$ **HIGH RISK**

### 4. Metrics & Diagnostics Engine (`src/evaluation.py`)
- Computes empirical evaluation metrics from held-out test predictions.
- Eliminates hardcoded or fabricated scores; metrics reflect exact model performance.
- Formulates per-class error profiles isolating false positives and false negatives.

### 5. Analyst Review & Visualization Dashboard (`src/visualizations.py`, `app.py`)
- Interactive interface rendering high-contrast, telemetry-styled charts via Plotly.
- Surfaces interactive confusion matrices, gauge indicators, feature importance rankings, and multi-axis performance radar charts.
- Provides guided navigation across 8 specialized security views.
