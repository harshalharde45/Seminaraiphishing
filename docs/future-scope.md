# Future Scope & Roadmap

## Planned Enhancements for Cyber Guardian AI

The following technical developments represent the future roadmap for evolving Cyber Guardian AI into an enterprise-grade platform:

### 1. Integration of Explainable AI (XAI)
- Incorporate **SHAP (SHapley Additive exPlanations)** and **LIME (Local Interpretable Model-agnostic Explanations)** to provide per-prediction feature attribution charts.
- Enable SOC analysts to inspect the exact feature values that triggered an alert for any individual flow.

### 2. Live Network Telemetry Capture
- Connect the preprocessing pipeline to native network interfaces via `pyshark` or `scapy`.
- Implement streaming feature extraction from raw `.pcap` feeds and real-time NetFlow/IPFIX records.

### 3. Deep Learning & Temporal Sequence Models
- Implement **Recurrent Neural Networks (LSTMs)** and **Temporal Convolutional Networks (TCNs)** to analyze session-level packet timing sequences.
- Address low-and-slow Advanced Persistent Threats (APTs) through stateful multi-step pattern detection.

### 4. Zero Trust & Automated SOAR Integration
- Integrate with **Security Orchestration, Automation, and Response (SOAR)** platforms (e.g., Shuffle, Cortex XSOAR) via authenticated REST webhooks.
- Dynamically trigger firewall quarantine rules or isolated VLAN assignments for high-confidence threats while adhering to Zero Trust principles.

### 5. Multi-Modal Threat Detection Modules
- Implement specialized sub-detectors for threats currently highlighted in the Educational section:
  - Deepfake audio/video detection using frequency-domain artifact analysis.
  - Phishing URL and email classification via Transformer-based NLP models.
  - LLM prompt-injection filters using semantic guardrail frameworks.

### 6. Continual & Active Learning
- Incorporate active learning loops where analyst validation feedback (confirming or dismissing alerts) automatically re-weights model priors and triggers scheduled retraining.
