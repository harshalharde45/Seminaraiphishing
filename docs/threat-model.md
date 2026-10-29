# Threat Model & Security Assessment

## System Scope & Assets
Cyber Guardian AI processes network flow telemetry to assist analysts in detecting security events.

### Primary Assets
1. **Analyst Decision Integrity**: Avoiding deception through manipulated alerts or poisoned models.
2. **System Availability**: Ensuring continuous responsiveness of the dashboard interface.
3. **Data Confidentiality & Privacy**: Protecting proprietary network flow attributes and preventing exfiltration.
4. **Model Trustworthiness**: Preventing adversarial evasion and parameter degradation.

---

## Threat Actors & Adversarial Profiles
- **External Adversary**: Attempts to craft network traffic that bypasses ML classification (adversarial perturbation).
- **Insider Threat / Untrusted User**: Uploads malformed, poisoned, or malicious CSV payloads to crash the server or induce model drift.
- **Resource Exhaustion Attacker**: Submits extremely large datasets or excessive requests to cause Denial of Service (DoS) on the Streamlit host.

---

## Trust Boundaries
1. **User Input Boundary**: Transition between uploaded raw CSV files and memory buffers.
2. **Data Pipeline Boundary**: Transition between raw tabular data and scikit-learn transformation objects.
3. **Model Prediction Boundary**: Transition from model outputs (uncalibrated probabilities) to analyst-facing risk labels.

---

## STRIDE Threat Analysis & Mitigations

| Category | Threat Description | Severity | Implemented Mitigation |
|---|---|---|---|
| **Spoofing** | Adversary alters packet headers to mimic benign protocols. | High | Multi-feature inspection incorporating host frequency counters and error rates rather than protocol type alone. |
| **Tampering** | Upload of poisoned training data containing inverted labels. | High | Validation checks enforcing minimum sample size and class distribution. Explicit disclosure that model training is local and ephemeral. |
| **Repudiation** | Analyst denies taking an action based on model findings. | Low | Educational prototype scope; in enterprise systems, full audit logging of analyst review must be maintained. |
| **Information Disclosure** | Leakage of private IPs or sensitive user identifiers. | Medium | Dataset schema strips IP addresses and payload contents, utilizing only non-identifying statistical flow metadata. |
| **Denial of Service** | Corrupted or massive CSV causing memory exhaustion / process crash. | High | Input validation rejects non-finite values, empty files, and malformed structures gracefully with user-facing alerts. |
| **Elevation of Privilege** | Code execution via malicious deserialization (`pickle`). | Critical | Model operates exclusively in-memory; no unsafe external pickle/joblib deserialization of untrusted uploaded files is allowed. |

---

## Operational Consequences of ML Classification Errors

### False Positives (Type I Error)
- **Definition**: Benign traffic classified as an attack.
- **Operational Impact**: Blocks legitimate business transactions, floods SOC queues with false alerts, and leads to analyst desensitization.
- **Mitigation**: Analyst review gate; risk confidence thresholds; balanced tree splitting.

### False Negatives (Type II Error)
- **Definition**: Malicious intrusion classified as benign traffic.
- **Operational Impact**: High risk of lateral movement, ransomware execution, or persistent data exfiltration.
- **Mitigation**: Defense-in-depth architecture; complementary signature IDS; continuous monitoring.
