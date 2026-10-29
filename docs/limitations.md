# System Limitations & Disclosures

## Academic & Engineering Disclosures

Cyber Guardian AI is designed as an educational and research prototype. In adherence to academic integrity and responsible AI principles, the following limitations must be noted:

### 1. Synthetic Benchmark Reliance
- The default demonstration dataset is generated programmatically using statistical random distributions.
- **Critical Caveat**: High classification accuracy on synthetic data does **not** indicate equivalent effectiveness on live enterprise networks. Real-world traffic exhibits severe non-linear noise, protocol variance, and zero-day evasion techniques absent from synthetic generators.

### 2. Tabular Flow Data vs. Deep Packet Inspection (DPI)
- The model evaluates aggregated statistical flow features (duration, packet counts, byte rates, port counters) rather than deep payload bytes.
- Consequently, application-layer payload exploits (e.g., SQL injection, advanced buffer overflows, steganography) cannot be detected without complementary DPI or WAF inspection.

### 3. Concept Drift & Distribution Shift
- The statistical properties of network traffic evolve constantly due to infrastructure upgrades, new user applications, and changing work habits. A static Random Forest model will suffer performance degradation over time unless retraining is performed.

### 4. Adversarial Evasion Vulnerability
- Machine learning models are inherently vulnerable to adversarial perturbation. Attackers who understand the model's feature importances can craft attacks (e.g., padding packet delays or modifying packet sizes) that force malicious flows into benign decision boundaries.

### 5. Independence of Records (Lack of Temporal Context)
- Each flow record is processed independently. The current prototype does not maintain stateful sequence tracking across hours or days, which is necessary to detect low-and-slow Advanced Persistent Threats (APTs).

### 6. Probability Calibration
- Decision forest probability estimates (`predict_proba`) represent tree ensemble voting proportions, not formally calibrated Bayesian posterior probabilities of an ongoing intrusion. They should be interpreted as relative confidence indicators rather than absolute risk values.
