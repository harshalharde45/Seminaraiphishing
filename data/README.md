# Dataset Directory

This directory is for network-traffic datasets.

## ⚠️ Synthetic Data Notice

By default, Cyber Guardian AI uses a **synthetically generated** dataset created by `src/data_processing.py`.

**The synthetic data:**
- Is programmatically generated using `numpy.random`
- Uses statistical distributions designed to make binary/multi-class traffic classification learnable
- Does NOT represent real network traffic
- Should NOT be used to claim real-world detection effectiveness

## Expected CSV Format

If you want to upload your own data, the CSV must contain these columns:

| Column | Type | Description |
|--------|------|-------------|
| `duration` | float | Connection duration in seconds |
| `protocol_type` | string | Protocol: `tcp`, `udp`, or `icmp` |
| `src_bytes` | float | Bytes sent from source |
| `dst_bytes` | float | Bytes sent from destination |
| `land` | int | 1 if src/dst host and port are the same |
| `wrong_fragment` | int | Number of wrong fragments |
| `urgent` | int | Number of urgent packets |
| `hot` | float | Number of hot indicators |
| `num_failed_logins` | int | Number of failed login attempts |
| `logged_in` | int | 1 if logged in successfully |
| `num_compromised` | float | Number of compromised conditions |
| `count` | float | Connections to same host in past 2 seconds |
| `srv_count` | float | Connections to same service in past 2 seconds |
| `same_srv_rate` | float | % of connections to same service |
| `diff_srv_rate` | float | % of connections to different services |
| `dst_host_count` | float | Connections to same destination host |
| `dst_host_srv_count` | float | Connections to same service on dest host |
| `dst_host_same_srv_rate` | float | % connections to same service on dest |
| `dst_host_diff_srv_rate` | float | % connections to different services on dest |
| `dst_host_same_src_port_rate` | float | % connections with same source port |
| `label` | string | Class label (e.g., `BENIGN`, `DoS`, `PortScan`, `BruteForce`, `DDoS`) |

> **Note:** The `label` column is required for training but optional for prediction-only mode.

## Compatible Public Datasets

The feature schema is inspired by the **KDD Cup 1999** and **NSL-KDD** dataset feature definitions.

If using NSL-KDD or a derivative:
- Rename columns to match the schema above
- Ensure label values match the expected format
- Document the source and any preprocessing applied

## What NOT to Commit

Do **not** commit:
- Raw packet captures (`.pcap` files)
- Personally identifiable network data
- Proprietary or restricted datasets
- Large files (>50 MB)

The `.gitignore` excludes `data/*.csv` and `data/*.pcap` to prevent accidental data commits.

