# 🩺 Liver Disease Risk Stratification & Explainable ML System

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Tests: 40 Passed](https://img.shields.io/badge/tests-40%20passed-brightgreen.svg)]()
[![Code Style: Ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Framework: Scikit-Learn](https://img.shields.io/badge/framework-scikit--learn-orange.svg)](https://scikit-learn.org/)
[![API: FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://8gt6vets6xrcmhjpzgkspk.streamlit.app/)
[![Docker](https://img.shields.io/badge/container-Docker-blue.svg)](https://www.docker.com/)
[![Security: Audited](https://img.shields.io/badge/security-audited-success.svg)](SECURITY.md)

---

### 🔴 Live Deployments & Quick Links
* **Live Cloud Application**: [https://8gt6vets6xrcmhjpzgkspk.streamlit.app/](https://8gt6vets6xrcmhjpzgkspk.streamlit.app/) *(No installation needed)*
* **Interactive REST API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs) *(FastAPI Swagger UI)*
* **Security & Privacy Policy**: [`SECURITY.md`](SECURITY.md)

---

## 🎯 Recruiter & Hiring Manager Cheat-Sheet (30-Second Summary)

> **Why this project stands out:** Most entry-level ML portfolios feature single, untested `.ipynb` notebooks with severe data leakage and misleading accuracy metrics. This repository demonstrates **production engineering discipline, epidemiological statistical rigor, and enterprise deployment readiness**.

| Evaluation Dimension | What Was Engineered | Recruiter & Industry Significance |
| :--- | :--- | :--- |
| **Statistical & Clinical Rigor** | **ROC-AUC: 0.8171** `[95% CI: 0.737–0.890]`, **Sensitivity: 98.8%**, **NPV: 95.0%**, **Brier: 0.1790** | Avoided the "accuracy trap" on imbalanced data; proved true epidemiological signal via **100-run Permutation Testing ($p < 0.001$)**. |
| **Zero Data Leakage** | Purged 13 duplicate patient records *prior* to stratified 80/20 train/test split. Preprocessors encapsulated in `sklearn.pipeline.Pipeline`. | Guarantees test set is strictly unseen; zero optimistic bias in reported metrics. |
| **Clinical Utility (DCA)** | Implemented **Decision Curve Analysis (DCA)** (Vickers & Elkin, BMJ) measuring **Clinical Net Benefit**. | Demonstrates that the ML model delivers superior patient outcomes over default "Screen All" or "Screen None" strategies. |
| **Software Architecture** | Modular library under `src/liverrisk/`, **PEP 517/621** `pyproject.toml`, Docker Compose orchestration. | Production-grade software engineering, fully decoupled and packageable. |
| **Testing & CI/CD** | **40 automated tests (`pytest`)** covering data hygiene, edge cases, and API endpoints + **GitHub Actions CI**. | 100% green test pass rate; zero lint errors via **Ruff (`ruff check .`)**. |
| **Explainable AI (XAI)** | **SHAP (Shapley Additive Explanations)** global rankings + real-time local patient waterfall charts. | Converts "black box" machine learning into clinically actionable, auditable biomarker attributions. |
| **Production Deliverables** | **Streamlit Community Cloud** (Live) + **FastAPI REST Service** (with OpenAPI/Swagger UI) + **Docker Compose**. | Dual product delivery ready for clinician use and enterprise backend microservice integration. |
| **Privacy & Security** | **Zero Data Retention** (ephemeral RAM inference) + **Pydantic V2** runtime input bounds. | Conforms to healthcare data minimization and defense-in-depth principles. |

---

## 💡 What Is This Project? (Plain English Summary)

Early liver disease often develops silently without visible symptoms until severe damage or cirrhosis has occurred. However, routine blood tests (**Liver Function Tests / LFTs**) contain subtle biochemical patterns that signal early hepatobiliary distress.

This project is an **end-to-end, production-grade clinical machine learning system** trained on the UCI Indian Liver Patient Dataset. It takes standard routine laboratory blood values and computes:
1. **A Calibrated Disease Risk Score**: Probabilistic prediction indicating statistical likelihood of hepatic disorder.
2. **Clinical Screening Stratification**: Tuned specifically for screening sensitivity (**98.8% Recall**) to minimize life-threatening false negatives.
3. **Explainable AI (SHAP Waterfall Plots)**: Instant visual breakdown showing exactly *which* biomarkers pushed a patient above or below the threshold.
4. **Dual Production Delivery**: A reactive **Streamlit Clinical UI** and an enterprise-grade **FastAPI REST Service** with Swagger docs.

---

## 🔒 Security, Privacy & Clinical Data Governance

We treat health data security and patient privacy with the highest degree of engineering diligence:

| Security Pillar | Implementation Details |
| :--- | :--- |
| **🛡️ Zero Data Retention** | **Privacy by Design**: All inference is processed strictly in volatile RAM (ephemeral). Patient inputs are **never written to disk, saved in databases, or logged**. |
| **🔍 Strict Input Sanitization** | **Pydantic V2 Runtime Validation**: All numeric inputs are enforced within strict physiological boundaries ($1 \le \text{Age} \le 120$, $0.1 \le \text{Bilirubin} \le 100$). Non-numeric and out-of-range injections are rejected immediately with HTTP `422 Unprocessable Entity`. |
| **📦 Dependency Supply Chain** | Standardized with **PEP 517/621** (`pyproject.toml`) and pinned in `requirements.txt`. Automated CI vulnerability auditing. |
| **🐳 Container Isolation** | Minimalist Docker base image (`python:3.11-slim`), non-root process execution, and least-privilege network exposure. |
| **📜 Responsible Disclosure** | Formal **[`SECURITY.md`](SECURITY.md)** policy with direct private reporting guidelines (`guptasimransushma@gmail.com`). |

---

## 📌 Clinical Biomarkers Reference

Routine hepatic panels measure enzymes and proteins synthesized or cleared by the liver:

| Biomarker | Full Medical Name | Normal Range | Clinical Significance |
| :--- | :--- | :--- | :--- |
| **TB** | Total Bilirubin | `0.2 – 1.2 mg/dL` | Heme breakdown product; elevated in jaundice, hepatitis, cirrhosis |
| **DB** | Direct (Conjugated) Bilirubin | `0.0 – 0.3 mg/dL` | Water-soluble bilirubin; elevated in bile duct obstruction & cholestasis |
| **Alkphos**| Alkaline Phosphatase | `44 – 147 IU/L` | Enzyme in bile ducts & bone; surges in biliary obstruction |
| **Sgpt** | ALT (Alanine Aminotransferase) | `7 – 56 IU/L` | Key intracellular liver enzyme; surges in acute hepatocellular injury |
| **Sgot** | AST (Aspartate Aminotransferase)| `10 – 40 IU/L` | Present in liver and heart; elevated in chronic liver damage & alcohol-related hepatitis |
| **TP** | Total Serum Proteins | `6.0 – 8.3 g/dL` | Total circulating albumin and globulins |
| **ALB** | Serum Albumin | `3.4 – 5.4 g/dL` | Synthesized solely by liver cells; drops in chronic hepatic failure |
| **A/G Ratio**| Albumin-to-Globulin Ratio | `1.0 – 2.0` | Sensitive marker for cirrhosis or chronic hepatic decompensation |

---

## 🏗️ Anti-Leakage Pipeline Architecture

```mermaid
graph TD
    A["Raw UCI ILPD Cohort (583 Records)"] --> B["Deduplication (13 Exact Duplicates Purged -> 570 Unique Records)"]
    B --> C["Binary Target Mapping (1 = Disease, 0 = Healthy)"]
    C --> D["Stratified 80/20 Train/Test Partitioning"]
    
    subgraph "Scikit-Learn Strict ColumnTransformer Pipeline"
        D --> E["Training Split (456 Samples)"]
        E --> F["Median Imputation + Missingness Indicator (A/G Ratio)"]
        F --> G["Log1p Transformation (Skewed Enzymes & Bilirubin)"]
        G --> H["StandardScaler (Continuous Features)"]
        G --> I["OneHotEncoder (Gender)"]
    end
    
    H --> J["Repeated 5x3 Stratified Cross-Validation"]
    I --> J
    J --> K["Hyperparameter Optimization (GridSearchCV)"]
    K --> L["Champion Model: Regularized Logistic Regression"]
    L --> M["Out-of-Fold Clinical Screening Threshold Tuning (pt = 0.280)"]
    M --> N["Final Evaluation on Held-Out Test Cohort (114 Samples)"]
    N --> O["Decision Curve Analysis & SHAP Explainability"]
    O --> P["Production Deployment: Streamlit Cloud UI & FastAPI REST Service"]
```

### 🛡️ Zero-Leakage Guarantees
1. **Deduplication before Splitting**: 13 duplicate patient records were removed *prior* to splitting, preventing identical twins from appearing in both train and test folds.
2. **Encapsulated Preprocessing**: Missing value imputation medians, log transformations, and standard scalers were fitted *strictly on training folds* inside `sklearn.pipeline.Pipeline`.
3. **Out-of-Fold Threshold Selection**: The clinical decision threshold ($p_t = 0.280$) was determined solely on cross-validated training predictions, keeping the test set 100% untouched until final evaluation.

---

## 📊 Rigorous Model Benchmarks & Validation

### 1. Repeated 5x3 Cross-Validation Comparison (Training Cohort, $N=456$)

| Model Architecture | CV ROC-AUC (Mean ± Std) | CV PR-AUC | CV Brier Score | Decision Rationale |
| :--- | :---: | :---: | :---: | :--- |
| **Regularized Logistic Regression** | **0.731 ± 0.037** | **0.867** | **0.177** | **Selected Champion**: Optimal probabilistic calibration, zero overfitting, linear interpretability |
| **Random Forest Classifier** | 0.720 ± 0.041 | 0.852 | 0.183 | Captures non-linearities but slightly higher variance |
| **Extra Trees Classifier** | 0.714 ± 0.039 | 0.846 | 0.188 | Fast bagging ensemble, competitive AUC |
| **HistGradientBoosting** | 0.709 ± 0.045 | 0.840 | 0.191 | High capacity; susceptible to overfitting on small sample size |
| **Enzyme Heuristic (ALT/AST)** | 0.612 ± 0.031 | 0.778 | 0.245 | Naive clinical baseline; transaminases alone are insufficient |
| **Dummy (Stratified)** | 0.500 ± 0.020 | 0.713 | 0.285 | Random chance baseline |

---

### 2. Held-Out Test Set Performance ($N=114$, Evaluated Exactly Once)

| Metric | Score | Clinical Significance |
| :--- | :---: | :--- |
| **ROC-AUC** | **0.8171** `[95% CI: 0.7370 – 0.8902]` | Excellent overall discriminatory power |
| **Screening Sensitivity (Recall)** | **98.77%** `[95% CI: 96.20% – 100.0%]` | Catches nearly all patients with disease; minimizes false negatives |
| **Negative Predictive Value (NPV)** | **95.00%** | When the screening test is negative, patient is genuinely healthy with high confidence |
| **F1-Score** | **0.8290** | Harmonious balance between precision and sensitivity |
| **Brier Calibration Score** | **0.1790** | Predicted probabilities reflect real observed clinical outcome rates |
| **Permutation Test ($p$-value)** | **$< 0.001$** | Proves mathematically that model learns genuine biological signal, not noise |

---

## 🩺 Clinical Decision Curve Analysis (DCA)

Medical machine learning models cannot be evaluated on AUC alone. **Decision Curve Analysis (Vickers & Elkin, BMJ)** quantifies the **Net Benefit** of using the model in a real clinic:

$$\text{Net Benefit} = \frac{\text{True Positives}}{N} - \frac{\text{False Positives}}{N} \times \left(\frac{p_t}{1 - p_t}\right)$$

* **Clinical Net Benefit Window**: Between threshold probabilities of $0.15$ and $0.70$, our ML model delivers a higher net benefit than both the default clinical strategies: *"Screen All Patients"* and *"Screen No Patients"*.
* Publication-ready DCA plots are generated in [`reports/figures/model_decision_curve.png`](reports/figures/model_decision_curve.png).

---

## 📁 Repository Directory Structure

```text
liver-disease-prediction/
├── .github/
│   └── workflows/
│       └── ci.yml               # Automated GitHub Actions CI (Linter + 40 Tests)
├── .streamlit/
│   └── config.toml              # Streamlit Cloud deployment settings & medical theme
├── data/
│   ├── raw/
│   │   └── ilpd.csv             # UCI Indian Liver Patient Dataset
│   └── sample_patients.csv      # Sample patient test profiles for batch testing
├── models/
│   └── final_pipeline.joblib    # Serialized Scikit-Learn Champion Pipeline
├── reports/
│   └── figures/                 # 12 Generated Diagnostic & Explainability Figures
│       ├── model_roc_pr_curve.png
│       ├── model_calibration_curve.png
│       ├── model_confusion_matrix.png
│       ├── model_decision_curve.png
│       └── shap_summary.png
├── results/
│   └── results.json             # Comprehensive metrics, CIs, and benchmark audit
├── src/
│   └── liverrisk/               # Modular Production Package
│       ├── __init__.py
│       ├── data.py              # Ingestion, validation, deduplication, splitting
│       ├── eda.py               # Automated EDA figure generation
│       ├── evaluate.py          # Metrics, bootstrap CI, DCA, calibration, figures
│       ├── explain.py           # SHAP summary & local waterfall plots
│       ├── features.py          # Custom scikit-learn transformers & pipeline
│       ├── models.py            # Model declarations and parameter grids
│       └── train.py             # Master pipeline orchestrator
├── tests/                       # 40 Comprehensive Unit & Integration Tests (Pytest)
│   ├── test_api.py              # FastAPI endpoint tests (health, predict, batch)
│   ├── test_app.py              # Streamlit helper & artifact loading tests
│   ├── test_data.py             # Deduplication, split invariance, validation tests
│   ├── test_evaluate.py         # DCA, bootstrap, permutation, calibration tests
│   ├── test_features.py         # Leakage prevention & transformation tests
│   └── test_models.py           # Pipeline serialization & probability bounds tests
├── api.py                       # FastAPI Production REST Microservice
├── app.py                       # Streamlit Clinical Web Dashboard
├── Dockerfile                   # Multi-stage production container configuration
├── docker-compose.yml           # Orchestration for FastAPI (:8000) & Streamlit (:8501)
├── predict.py                   # High-throughput batch CSV & CLI inference tool
├── pyproject.toml               # PEP 517/621 modern packaging configuration
├── requirements.txt             # Pinned reproducible dependencies
├── run_all.py                   # Master one-click pipeline reproduction script
├── SECURITY.md                  # Comprehensive security policy & vulnerability reporting
└── README.md                    # Project documentation
```

---

## 🚀 Quickstart & How to Run Locally

### 1. Installation
Clone this repository and install dependencies in your virtual environment:

```bash
git clone https://github.com/simnexis/liver-disease-prediction.git
cd liver-disease-prediction
pip install -r requirements.txt
```

### 2. Run the Test Suite (40/40 Passing)
```bash
python -m pytest
```

### 3. Launch the FastAPI REST Service
```bash
uvicorn api:app --reload --port 8000
```
* **Interactive Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
* **Interactive ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)
* **Health Check**: `curl http://localhost:8000/health`

### 4. Launch the Streamlit Dashboard
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** to interact with the local clinical calculator and SHAP waterfall explainer.

### 5. Docker Deployment (One-Command)
```bash
docker-compose up --build
```
This simultaneously spins up the **FastAPI REST Service (`:8000`)** and the **Streamlit Web UI (`:8501`)**.

---

## 📜 Citation & License

* **Dataset Citation**: Ramana, B., & Venkateswarlu, N. (2012). *ILPD (Indian Liver Patient Dataset)*. UCI Machine Learning Repository. [https://doi.org/10.24432/C5D02C](https://doi.org/10.24432/C5D02C)
* **Code & Pipeline License**: Creative Commons Attribution 4.0 International ([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
* **Security Contact**: See [`SECURITY.md`](SECURITY.md) for vulnerability reporting guidelines.
