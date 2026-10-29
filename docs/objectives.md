# Research & Engineering Objectives

## Primary Objective
Design, implement, validate, and document **Cyber Guardian AI**, an educational and research-oriented machine-learning framework for network threat classification, risk evaluation, and analyst review.

## Detailed Engineering Objectives

1. **Robust Data Validation and Preprocessing Architecture**:
   - Establish strict schema validation for network connection records.
   - Centralize data cleaning (coercing invalid values, handling missing fields, resolving infinite/NaN floating-point anomalies).
   - Prevent data leakage between training and evaluation phases.

2. **Machine Learning Pipeline Implementation**:
   - Construct a modular scikit-learn classification pipeline combining numerical scaling (`StandardScaler`), categorical one-hot encoding (`OneHotEncoder`), and a balanced Random Forest ensemble (`RandomForestClassifier`).
   - Implement reproducible, stratified cross-validation/splits preserving class distributions.

3. **Multi-Vector Threat Discrimination**:
   - Train the classifier to distinguish normal operational network traffic (`BENIGN`) from prevalent attack classes, including Denial of Service (`DoS`), Distributed Denial of Service (`DDoS`), Network Reconnaissance (`PortScan`), and Authentication Brute Force (`BruteForce`).

4. **Rigorous, Authentic Model Evaluation**:
   - Compute real evaluation metrics directly from held-out test predictions: Accuracy, Macro/Weighted Precision, Macro/Weighted Recall, and Macro/Weighted F1-scores.
   - Generate full confusion matrices and granular false-positive (FP) and false-negative (FN) breakdown tables.
   - Derive global feature importances to inspect model decision drivers.

5. **Professional Cybersecurity Analyst Dashboard**:
   - Provide an interactive, responsive Streamlit interface with dark telemetry aesthetics.
   - Incorporate modular views: Overview Dashboard, Data Explorer, AI Detection Lab, Threat Analysis, Security Insights, Model Evaluation, and Educational Threat Intelligence.

6. **Responsible Disclosure & Academic Integrity**:
   - Prominently disclose the use of synthetic benchmark data.
   - Document systemic limitations, dataset shifts, and false-positive/negative risks.
   - Emphasize the vital role of human analyst verification in operational security.
