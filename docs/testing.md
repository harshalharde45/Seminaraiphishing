# Testing and Quality Assurance

## Testing Strategy
The testing suite for **Cyber Guardian AI** covers unit functionality, data validation boundaries, machine learning pipeline integrity, and end-to-end integration workflows. All automated tests are executed using **Pytest (v8.3.3)**.

---

## Test Execution Summary

```bash
python -m pytest tests/ -v --tb=short
```

- **Total Test Cases**: 71
- **Passed**: 71
- **Failed**: 0
- **Duration**: ~40 seconds
- **Platform**: Windows / Python 3.12.10

---

## Breakdown of Test Suites

### 1. Data Processing Tests (`tests/test_data_processing.py` — 26 tests)
- **Synthetic Data Generation**:
  - Verification of expected column schemas and data types.
  - Verification of class distribution and minimum instance counts.
  - Verification of deterministic output with fixed random seeds.
  - Absence of null/NaN values in synthetic datasets.
- **Validation Engine**:
  - Rejection of empty DataFrames and NoneType objects.
  - Rejection of datasets missing mandatory flow features.
  - Rejection of datasets missing the target `label` column when required.
  - Rejection of single-class datasets for training.
  - Enforcement of sample size thresholds.
- **CSV Ingestion**:
  - Successful parsing of compliant CSV byte buffers.
  - Graceful rejection of malformed binary strings without process termination.
  - Trimming and sanitization of whitespace in column headers.
- **Data Preprocessing**:
  - Coercion of corrupted numeric strings into floats.
  - Imputation of missing numeric values with medians.
  - Safe replacement of infinite values (`inf`, `-inf`).

### 2. Model Pipeline Tests (`tests/test_model.py` — 16 tests)
- Pipeline construction with `ColumnTransformer`, `StandardScaler`, and `OneHotEncoder`.
- Stratified 75/25 train-test split verification.
- Reproducibility of model weights and predictions under identical random seeds.
- Predict output shape and class probability bounds ($\sum P = 1.0$).
- Feature importance extraction and sum-to-one validation.

### 3. Evaluation Tests (`tests/test_evaluation.py` — 14 tests)
- Perfect classifier metric computation (Accuracy = 1.0, F1 = 1.0).
- Degraded/inverted classifier evaluation.
- Multiclass confusion matrix shape and consistency.
- Correct calculation of true positives, false positives, false negatives, and true negatives per class.
- Safe handling of edge cases (single-class evaluation warnings, zero division guards).

### 4. End-to-End Detection Workflow Tests (`tests/test_detection.py` — 15 tests)
- Full orchestration pipeline execution (`run_training` $\rightarrow$ `run_evaluation` $\rightarrow$ `run_prediction`).
- Integration check on unlabelled batch prediction.
- Validation of risk level assignment (`LOW` vs. `HIGH`) and confidence scores.
- Summary metric calculations (aggregate counts, percentage consistency).
