# Viva Voce Questions & Answers

### 1. What role does AI play in modern cybersecurity?
**Answer**: AI assists defenders by automating high-volume telemetry analysis, detecting anomalous patterns, prioritizing alerts, and reducing incident response times. Concurrently, adversaries misuse AI to automate reconnaissance, craft polymorphic malware, generate targeted phishing emails, and conduct automated credential attacks.

### 2. How does Cyber Guardian AI classify network traffic?
**Answer**: It uses a supervised Random Forest ensemble classifier trained on tabular network flow features (e.g., flow duration, byte counts, protocol types, and host connection rates) to categorize each connection as `BENIGN` or an attack type (`DoS`, `DDoS`, `PortScan`, `BruteForce`).

### 3. Why was Random Forest selected instead of a Deep Neural Network?
**Answer**: Random Forest is highly effective on tabular structured data, robust to outliers, offers native feature importance interpretability, trains rapidly without specialized GPU hardware, and provides strong non-linear decision boundaries with lower risk of overfitting on small-to-medium datasets.

### 4. What is the difference between training data and testing data?
**Answer**: Training data is used by the algorithm to learn feature-to-class relationships and optimize internal split criteria. Testing data is strictly held out and unseen during training to evaluate how well the model generalizes to new, unseen records.

### 5. Why is stratified sampling important in your train/test split?
**Answer**: Network traffic datasets often suffer from severe class imbalance (e.g., benign traffic vastly outnumbers attack types). Stratified sampling guarantees that both the training and testing sets retain the exact proportional representation of each class.

### 6. What preprocessing steps are executed before passing data to the model?
**Answer**: Features are separated into numeric and categorical types using a scikit-learn `ColumnTransformer`. Numeric features undergo missing-value median imputation and standard scaling (`StandardScaler`). Categorical features (`protocol_type`) are encoded via `OneHotEncoder(handle_unknown='ignore')`.

### 7. What is data leakage, and how is it prevented in this project?
**Answer**: Data leakage occurs when information from outside the training dataset (such as test set distributions or the target label itself) is inadvertently shared with the model during training. We prevent this by encapsulating all scaling and transformations inside a scikit-learn `Pipeline` that fits exclusively on the training split.

### 8. Why is Accuracy alone an insufficient metric in cybersecurity?
**Answer**: If a dataset contains 99% benign traffic and 1% attacks, a naive model predicting 100% benign achieves 99% accuracy while failing to detect a single attack. Metrics such as Precision, Recall, and Macro-averaged F1-score are necessary to measure detection efficacy across all classes.

### 9. What is the difference between Precision and Recall?
**Answer**: 
- **Precision** ($\frac{TP}{TP + FP}$): Of all connections the model flagged as attacks, what percentage were actual attacks? High precision minimizes false alarms.
- **Recall** ($\frac{TP}{TP + FN}$): Of all actual attacks occurring on the network, what percentage did the model catch? High recall minimizes missed intrusions.

### 10. What does the F1-Score represent?
**Answer**: The F1-Score is the harmonic mean of Precision and Recall ($2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$). It provides a single balanced metric, especially under unequal class distributions. Macro-averaged F1 computes this score independently for each class and averages them equally.

### 11. What is a Confusion Matrix, and how is it read?
**Answer**: A confusion matrix is an $N \times N$ grid comparing ground-truth classes (rows) against predicted classes (columns). Diagonal cells represent correct classifications ($TP$ and $TN$), whereas off-diagonal elements indicate misclassifications (false positives and false negatives).

### 12. In cybersecurity, which is more dangerous: a False Positive or a False Negative?
**Answer**: A **False Negative** is typically far more dangerous because an active attack bypasses detection and establishes persistence or executes payloads. However, excessive **False Positives** disrupt legitimate business operations and cause severe analyst alert fatigue.

### 13. What is the difference between synthetic and real-world network data?
**Answer**: Synthetic data is programmatically generated using statistical approximations and mathematical distributions. Real-world traffic contains protocol anomalies, packet fragmentation, temporal dependencies, user idiosyncrasies, and zero-day evasion tactics not present in synthetic generators.

### 14. Why must synthetic data results be disclosed clearly?
**Answer**: Academic and engineering integrity requires transparent reporting. High classification performance on synthetic data does not prove that an algorithm will succeed in enterprise production environments.

### 15. How does Cyber Guardian AI handle missing or malformed values?
**Answer**: The data ingestion layer coerces non-numeric strings to NaN, substitutes infinite floats, and imputes missing numeric values with the training set median, while missing categorical values are imputed with the most frequent value.

### 16. What is class imbalance, and how does your model address it?
**Answer**: Class imbalance occurs when one class (typically benign) dominates the dataset. The model mitigates this by applying `class_weight='balanced'`, which automatically penalizes misclassifications on minority attack classes inversely proportional to their frequencies.

### 17. How is feature importance derived in Random Forest?
**Answer**: Feature importance is calculated using Gini impurity reduction (Mean Decrease in Impurity). It aggregates the total reduction in split criterion impurity brought by a given feature across all decision trees in the ensemble.

### 18. Why does Cyber Guardian AI avoid automated blocking or isolation?
**Answer**: Fully autonomous blocking based on uncalibrated statistical models risks terminating critical legitimate network operations. A "Human-in-the-Loop" architecture empowers human analysts to verify high-risk alerts before applying policy enforcements.

### 19. What trust boundaries exist in the application?
**Answer**: Trust boundaries exist at the user CSV upload interface, the input transformation boundary into scikit-learn arrays, and the interpretation boundary where probabilistic scores become analyst-facing risk labels.

### 20. How does the system defend against unsafe code execution via file upload?
**Answer**: The application parses uploaded files exclusively as raw CSV tabular data via Pandas. It does not permit arbitrary Python execution, shell execution, or unsafe object deserialization (`pickle.load`).

### 21. What is Concept Drift in network traffic analysis?
**Answer**: Concept drift occurs when the statistical properties of network traffic and cyber attack patterns change over time, rendering historical models inaccurate unless periodically retrained with fresh telemetry.

### 22. What are the key limitations of Cyber Guardian AI?
**Answer**: Key limitations include reliance on synthetic demonstration data, lack of stateful temporal tracking for multi-stage attacks, inability to inspect encrypted application-layer payloads without DPI, and sensitivity to adversarial traffic evasion.

### 23. What future enhancements are planned for Cyber Guardian AI?
**Answer**: Planned improvements include live packet capture integration via PyShark, Explainable AI (SHAP/LIME) for local sample explanations, temporal sequence modeling via LSTMs, and automated Zero Trust SOAR webhook dispatching.
