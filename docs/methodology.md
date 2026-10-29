# Methodology

## Scientific & Engineering Methodology

Cyber Guardian AI follows a structured 7-stage research and implementation pipeline:

```
[1. Problem Study]
        │
[2. Literature Survey]
        │
[3. Requirements & Technology Analysis]
        │
[4. Framework Development (Cyber Guardian AI)]
        │
[5. Data Collection, Generation & Preprocessing]
        │
[6. Empirical Model Training & Evaluation]
        │
[7. Analyst Review, Validation & Future Scope]
```

---

## Detailed Pipeline Phases

### Phase 1: Problem Study
- Investigated the operational challenges of modern intrusion detection systems (SOC alert fatigue, high volume of encrypted traffic, evasive threats).
- Identified the limitations of static heuristic rules and uncalibrated ML systems.

### Phase 2: Literature Survey
- Synthesized findings across 5 primary domains:
  1. **AI for Network Cybersecurity**: ML algorithms for packet flow anomaly detection.
  2. **AI-Generated Phishing**: LLM-driven credential harvesting.
  3. **Deepfake Threats**: Audio/video identity cloning in enterprise environments.
  4. **AI-Assisted Malware**: Polymorphic code generation and evasion strategies.
  5. **Prompt Injection & LLM Vulnerabilities**: Indirect adversarial instructions.

### Phase 3: Requirements & Technology Analysis
- Selected a local, reproducible Python 3.12 stack avoiding third-party cloud dependencies or external paid APIs:
  - **Streamlit (v1.40.0)** for user interface.
  - **Scikit-learn (v1.5.2)** for pipelines, transformers, and model ensembles.
  - **Pandas (v2.2.3)** and **NumPy (v2.5.4)** for tabular computation.
  - **Plotly (v5.24.1)** and **Matplotlib (v3.9.2)** for visualization.
  - **Pytest (v8.3.3)** for automated regression and integration testing.

### Phase 4: Framework Architecture Design
- Formulated the **Cyber Guardian AI** architecture:
  - Decoupled modules for data handling, model training, evaluation, and visualization.
  - Centralized schema definitions adhering to industry flow features (KDD/NSL-KDD schema alignment).

### Phase 5: Data Generation & Preprocessing
- Developed a synthetic benchmark generator (`generate_synthetic_dataset`) producing 1,000+ records across 5 distinct classes (`BENIGN`, `DoS`, `DDoS`, `PortScan`, `BruteForce`).
- Engineered data sanitization procedures:
  - String stripping and index-column dropping.
  - Type coercion and non-finite value replacement.
  - Safe median/mode imputation within scikit-learn transformers.

### Phase 6: Model Training & Evaluation
- Applied stratified sampling to ensure consistent representation of minority classes.
- Fitted a 150-estimator Random Forest classifier with `class_weight='balanced'`.
- Evaluated predictions exclusively against the held-out 25% test partition.

### Phase 7: Verification & Feedback
- Implemented automated test coverage across 71 unit and integration test cases.
- Validated error boundaries, shape consistency, probability calibrations, and missing-feature fallbacks.
