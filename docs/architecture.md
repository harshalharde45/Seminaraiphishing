# System Architecture

## Architectural Principles
Cyber Guardian AI adheres to core software engineering principles:
1. **Modularity**: Separation of concerns across data parsing, model lifecycle, evaluation, and UI presentation.
2. **Stateless Operations**: Core ML algorithms operate deterministically given an input dataset and random seed.
3. **Defense-in-Depth**: Strict input validation prior to passing data to transformation pipelines.
4. **Data Isolation**: Complete decoupling between training, validation, testing, and unlabeled batch prediction workflows.

---

## Architectural Diagram

```mermaid
flowchart TD
    User([Security Analyst]) --> UI[Streamlit Frontend Dashboard]

    subgraph Data Layer
        UI -->|Generate Demo| SynGen[Synthetic Generator]
        UI -->|Upload CSV| CSVParser[CSV Ingestion Engine]
        SynGen --> Preproc[Data Preprocessing & Cleaning]
        CSVParser --> Preproc
        Preproc --> Validator{Schema Validator}
        Validator -->|Invalid| UIError[User Warning / Schema Guidance]
        Validator -->|Valid| FeatureEngine[Feature Separation & Encoding]
    end

    subgraph ML Pipeline
        FeatureEngine --> Splitter[Stratified 75/25 Train-Test Split]
        Splitter --> XTrain[Training Partition]
        Splitter --> XTest[Held-out Test Partition]
        XTrain --> Scaler[StandardScaler + OneHotEncoder]
        Scaler --> RF[Random Forest Ensemble (150 Estimators)]
        RF --> FittedPipeline[Fitted Pipeline]
    end

    subgraph Evaluation & Prediction
        FittedPipeline --> Evaluator[Evaluation Engine]
        XTest --> Evaluator
        Evaluator --> Metrics[Empirical Metrics & Confusion Matrix]
        
        UI -->|New Records| BatchPredict[Prediction Engine]
        BatchPredict --> FittedPipeline
        BatchPredict --> RiskScorer[Risk & Confidence Assignment]
    end

    subgraph Visual Analytics
        Metrics --> DashView[Model Evaluation Dashboard]
        RiskScorer --> ThreatView[Threat Analysis Dashboard]
        DashView --> UI
        ThreatView --> UI
    end
```

---

## Component Specifications

| Module | Core Responsibility | Key Classes / Functions |
|---|---|---|
| `app.py` | UI layout, state orchestration, routing | `main()`, `page_overview()`, `page_detection_lab()` |
| `src/data_processing.py` | Validation, cleaning, synthetic data creation | `generate_synthetic_dataset()`, `validate_dataframe()`, `preprocess_dataframe()` |
| `src/model.py` | Pipeline assembly, training, persistence | `build_pipeline()`, `train_model()`, `predict()`, `get_feature_importance()` |
| `src/detection.py` | Workflow orchestration, batch risk scoring | `run_training()`, `run_evaluation()`, `run_prediction()`, `get_threat_summary()` |
| `src/evaluation.py` | Metric computation, classification reports | `compute_metrics()`, `format_per_class_metrics()`, `interpret_metrics()` |
| `src/visualizations.py` | Plotly & Matplotlib dashboard charts | `plot_confusion_matrix()`, `plot_risk_gauge()`, `plot_per_class_metrics()` |
