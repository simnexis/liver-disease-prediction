# Security Policy & Clinical Data Governance

We take the security of this clinical machine learning software and patient privacy with the highest degree of diligence.

---

## 🔒 Supported Versions

The following versions of this project currently receive security updates and dependency patches:

| Version | Supported | Status |
| :--- | :---: | :--- |
| `1.0.x` | ✅ Yes | Active Long-Term Support (LTS) |
| `< 1.0.0` | ❌ No | Deprecated development prototypes |

---

## 🛡️ Security Architecture & Privacy Principles

This system has been built from the ground up according to **Privacy by Design** and **Defense in Depth**:

### 1. Zero Data Retention (In-Memory Ephemeral Inference)
* **No Persistent Storage of Health Data**: Patient biomarker inputs submitted via the Streamlit web dashboard or the FastAPI REST service are processed strictly in RAM and discarded immediately after response transmission.
* **No Telemetry or Logging of PHI**: No Protected Health Information (PHI), patient identifiers, blood panels, or IP addresses are logged to disk, database, or third-party analytics.

### 2. Strict Input Validation & Denial-of-Service Defense
* **Pydantic V2 Boundary Enforcement**: All REST endpoints enforce rigid type validation and physiological boundary constraints (e.g., $1 \le \text{Age} \le 120$, $0.1 \le \text{Total Bilirubin} \le 100$).
* **Payload Sanitation**: Non-numeric injection, negative values, and out-of-distribution payloads are rejected with HTTP `422 Unprocessable Entity` before reaching model execution steps.
* **Maximum Upload Limits**: Streamlit and FastAPI limit payload and file upload sizes to prevent memory exhaustion attacks.

### 3. Supply Chain Security & Dependency Auditing
* Dependencies in `requirements.txt` and `pyproject.toml` are pinned and audited against the Python Packaging Advisory Database.
* Continuous Integration via GitHub Actions automatically executes static security analysis and unit regression suites on all commits.

### 4. Container Isolation & Least Privilege
* The provided `Dockerfile` runs on minimal, stripped-down Linux base images (`python:3.11-slim`) without extraneous system utilities or root-level daemons.
* Docker Compose exposes only designated non-privileged ports (`:8000` for REST API, `:8501` for Streamlit UI).

---

## 🚨 Reporting a Vulnerability

If you discover a security vulnerability or potential data privacy concern, **please DO NOT create a public GitHub issue**.

Please report vulnerabilities privately:

1. **Email**: Send detailed information to the project maintainer at **`guptasimransushma@gmail.com`**.
2. **Details to Include**:
   * Type of issue (e.g., buffer handling, dependency vulnerability, input bypass, unexpected data leak).
   * Step-by-step instructions or minimal proof-of-concept to reproduce the behavior.
   * Potential impact or attack vector.
3. **Response Timeline**:
   * **Acknowledgment**: Within **24 hours**.
   * **Initial Assessment & Severity Scoring**: Within **48 hours**.
   * **Remediation & Patch Deployment**: Within **5 business days** for high-severity findings.

We follow Responsible Disclosure practices and will credit security researchers who report issues responsibly.

---

## ⚖️ Regulatory & Educational Disclaimer

This project is an **academic and research demonstration** of classical machine learning and epidemiological risk stratification. It is **not** an FDA-cleared, CE-marked medical device and is not intended to provide clinical diagnoses, direct patient management, or replace certified clinical laboratory diagnostics.
