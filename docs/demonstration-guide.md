# Demonstration Guide

## 1. Executive Summary
- **Problem**: Modern cybersecurity teams face overwhelming alert fatigue and struggle against dynamic, automated network attacks that evade static signatures.
- **Why AI/ML**: Machine learning algorithms detect complex non-linear statistical anomalies and correlate high-throughput flow metrics faster than manual triage.
- **What Cyber Guardian AI Implements**: An end-to-end network traffic classification pipeline using Random Forest ensembles, transparent preprocessing, stratified validation, and interactive analyst dashboards.

---

## 2. Step-by-Step Live Demonstration Walkthrough

### Step 1: Start the Dashboard
Open your terminal in the project directory and execute:
```powershell
streamlit run app.py
```
The browser will automatically open at `http://localhost:8501`.

### Step 2: Overview Dashboard
- Point out the dark cybersecurity UI designed for SOC analysts.
- Highlight the **Educational Prototype Notice** stating that demonstration data is synthetic.
- Click **"🔬 Load Demo Dataset"**.
- View the real-time calculated Dataset Summary (1,000+ records, 21 columns, class distribution).

### Step 3: Data Explorer
- Navigate to **"🔍 Data Explorer"** via the sidebar.
- Inspect the data preview table.
- Open the **"📈 Class Distribution"** tab to view the horizontal bar chart and donut chart showing proportions of `BENIGN`, `DoS`, `DDoS`, `PortScan`, and `BruteForce`.
- Switch to the **"✅ Validation"** tab to showcase automated schema checks and null-value handling.

### Step 4: AI Detection Lab (Training & Evaluation)
- Navigate to **"🧪 AI Detection Lab"**.
- Click **"🚀 Train Classifier"**.
- Observe the training progression:
  - Scikit-learn `Pipeline` applies `ColumnTransformer`, standardizing numeric features and one-hot encoding protocols.
  - Stratified 75/25 split creates an uncorrupted held-out test partition.
  - Random Forest fits 150 decision trees with balanced class weights.
- View immediate training metadata: Sample counts, test counts, and initial macro metrics.

### Step 5: Threat Analysis
- Switch to **"⚠️ Threat Analysis"**.
- Examine the **Suspicious Traffic Risk Gauge** (displaying overall attack percentage).
- Review the **Predicted Traffic Classification** bar chart.
- Inspect individual records with their predicted label, risk category (`LOW` vs. `HIGH`), and confidence score.
- Review the **Confusion Matrix** and per-class False Positive / False Negative counts.

### Step 6: Security Insights & Feature Importance
- Navigate to **"💡 Security Insights"**.
- Review computed observations automatically derived from model outputs.
- Inspect the **Top Feature Importances** chart (e.g., `count`, `dst_bytes`, `dst_host_count`) explaining which features drive the model's decisions.

### Step 7: Model Evaluation & Limitations
- Go to **"📊 Model Evaluation"**.
- Examine the **Model Performance Radar** and per-class Precision/Recall/F1 table.
- Emphasize the **Known Limitations** section: Explain why synthetic data performance must never be conflated with real-world security efficacy.

### Step 8: Educational Threat Intelligence & About
- Conclude by browsing the **"📚 Threat Intelligence"** section covering AI-generated phishing, deepfakes, malware, and prompt injection as future research areas.
- Highlight the project group credits under **"ℹ️ About"** (Group: Cyber Nexus).
