# 🏥 Clinical EHR Prediction Dashboard

> **An automated machine learning pipeline with continual learning to predict clinical conditions from Electronic Health Records — built to detect, adapt to, and overcome temporal data drift.**

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Pipeline-F7931E?style=flat-square&logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Data Dictionary](#-data-dictionary)
- [ML Pipeline](#-ml-pipeline)
- [Getting Started](#-getting-started)
- [Running the Dashboard](#-running-the-dashboard)
- [Results & Performance](#-results--performance)
- [Continual Learning Strategy](#-continual-learning-strategy)
- [Tech Stack](#-tech-stack)
- [Contributing](#-contributing)

---

## 🔍 Overview

The **Clinical EHR Prediction Dashboard** is a production-ready machine learning system that predicts clinical conditions — such as **Hypertension** and **Diabetes** — from synthetic Electronic Health Records (EHR). It is built on the **Synthea-MIMIC** simulated hospital dataset.

The core challenge this project addresses is **Temporal Data Drift**: as patient populations, treatment practices, and billing patterns evolve over time, a model trained on historical data will progressively degrade in performance on newer records. Most healthcare ML systems simply retrain from scratch — an expensive, time-consuming process.

This system solves it differently.

By chronologically splitting the data into **Historical** and **Current** cohorts, detecting drift across regimes, and applying **Continual Learning (CL)** techniques, the pipeline incrementally updates models on new data streams — restoring and often **exceeding original performance** without full retraining.

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🔄 **Temporal Drift Detection** | Chronological train/test splitting to simulate real-world regime shifts |
| 🧠 **Continual Learning** | Incremental model updates via `partial_fit` (SVM/MLP) without catastrophic forgetting |
| 📊 **Interactive Dashboard** | Full Streamlit UI for EDA, drift visualization, and model monitoring |
| ⚖️ **Class Imbalance Handling** | `class_weight='balanced'` for robust performance on skewed medical data |
| 🎯 **Threshold Tuning** | F1-score–optimized decision boundary calibration per model |
| 🗄️ **Artifact Persistence** | Trained models saved/loaded via `joblib` for reproducible experiments |
| 📈 **Multi-Model Benchmarking** | Side-by-side comparison of Decision Tree, SVM, and MLP Neural Network |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   CLINICAL EHR PIPELINE                     │
│                                                             │
│  ┌────────────┐    ┌─────────────────┐    ┌─────────────┐  │
│  │  Raw CSVs  │───▶│ data_processing │───▶│  Features   │  │
│  │ (synthea)  │    │     .py         │    │  DataFrame  │  │
│  └────────────┘    └─────────────────┘    └──────┬──────┘  │
│                                                  │         │
│                          ┌───────────────────────┤         │
│                          │                       │         │
│                   ┌──────▼──────┐        ┌───────▼──────┐  │
│                   │  Historical │        │   Current    │  │
│                   │  Cohort     │        │   Cohort     │  │
│                   │  (pre-2018) │        │  (post-2018) │  │
│                   └──────┬──────┘        └───────┬──────┘  │
│                          │                       │         │
│                   ┌──────▼──────┐                │         │
│                   │  Baseline   │◀───── Drift ───┘         │
│                   │  Training   │       Detection          │
│                   └──────┬──────┘                          │
│                          │                                 │
│                   ┌──────▼──────────────────────────┐      │
│                   │     Continual Learning Update   │      │
│                   │  SVM/MLP: partial_fit()         │      │
│                   │  Decision Tree: Merge & Refit   │      │
│                   └──────┬──────────────────────────┘      │
│                          │                                 │
│                   ┌──────▼──────┐                          │
│                   │  Streamlit  │                          │
│                   │  Dashboard  │                          │
│                   └─────────────┘                          │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
clinical-ehr-dashboard/
│
├── 📂 src/
│   ├── data_processing.py          # Feature engineering, merging, temporal split
│   └── models.py                   # ModelPipeline class (DT, SVM, MLP)
│
├── 📂 synthea-mimic/
│   └── csv/
│       ├── patients.csv
│       ├── encounters.csv
│       ├── conditions.csv
│       ├── observations.csv
│       ├── medications.csv
│       ├── procedures.csv
│       ├── claims.csv
│       └── ... (see Data Dictionary)
│
├── 📂 artifacts/                   # Saved model states (joblib)
│   ├── decision_tree.pkl
│   ├── svm_model.pkl
│   └── mlp_model.pkl
│
├── TeamXX_Assignment2_dashboard.py # Streamlit dashboard entrypoint
├── requirements.txt
└── README.md
```

---

## 🗄️ Data Dictionary

The `synthea-mimic/csv/` directory contains simulated hospital data. All tables are aggregated to the **Encounter level** during preprocessing.

| File | Description |
|---|---|
| `patients.csv` | Demographics: Birthdate, Race, Ethnicity, Gender, Income |
| `encounters.csv` | **Central table.** Each row = one clinic visit or hospital admission |
| `conditions.csv` | Diagnoses; used to generate the binary `TARGET` label (e.g., Hypertension) |
| `observations.csv` | Clinical vitals & labs (BMI, Systolic/Diastolic BP, Heart Rate) — aggregated as mean & variance per encounter |
| `medications.csv` | Medication logs — aggregated into counts and total cost |
| `procedures.csv` | Clinical procedures — aggregated into counts and cost sums |
| `immunizations.csv` | Immunization records per encounter |
| `supplies.csv` | Supply usage and costs |
| `careplans.csv` | Care plan assignments |
| `devices.csv` | Device usage records |
| `imaging_studies.csv` | Imaging study logs |
| `organizations.csv` | Treating hospital metadata (Revenue, Utilization) |
| `providers.csv` | Practitioner info (Specialty, Gender) |
| `payer_transitions.csv` | Insurance transitions per patient |
| `claims.csv` | Billing and insurance claim data |
| `claims_transactions.csv` | Itemized transaction records (outstanding amounts, total cost) |

---

## 🤖 ML Pipeline

### Stage 1 — Temporal Splitting & Drift Detection

The dataset is split at a chronological threshold (default: **2018**):

- **Regime 1 (Historical)**: Pre-2018 records → Training & baseline evaluation
- **Regime 2 (Current)**: Post-2018 records → Drift evaluation & continual learning

Drift is measured as shifts in:
- Disease **prevalence rates**
- Distributions of numerical features (Age, Billing Costs, BMI)
- Model performance degradation (ΔF1, ΔAUC between regimes)

### Stage 2 — Baseline Model Training

Three classifiers are trained on historical data:

| Model | Notes |
|---|---|
| **Decision Tree** | Gini-based, class-weight balanced |
| **SVM (SGD)** | Linear SVM via `SGDClassifier`; supports `partial_fit` |
| **MLP Neural Network** | Multi-layer perceptron; supports `partial_fit` |

All models use `class_weight='balanced'` to handle clinical data imbalance.

**Metrics tracked:** ROC-AUC, F1-Score, Precision, Recall, Accuracy

### Stage 3 — Degradation Evaluation

Baseline models are applied directly to the Current dataset. Performance drop (drift impact) is visualized via:
- Side-by-side **Confusion Matrices**
- **ROC Curve** comparisons
- **Bar charts** of metric deltas across regimes

### Stage 4 — Continual Learning Update

| Model | Update Strategy |
|---|---|
| SVM | `partial_fit` on new data stream |
| MLP | `partial_fit` on new data stream |
| Decision Tree | Merge historical + current subsets, refit |

Post-update **threshold tuning** recalibrates classification boundaries for optimal F1.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/clinical-ehr-dashboard.git
cd clinical-ehr-dashboard

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Linux/macOS
venv\Scripts\activate         # Windows

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```
streamlit
scikit-learn
pandas
numpy
matplotlib
seaborn
joblib
scipy
```

---

## 📊 Running the Dashboard

```bash
streamlit run TeamXX_Assignment2_dashboard.py
```

The dashboard will open at `http://localhost:8501` and provides:

- **Pipeline Configuration** — Set the temporal split threshold and target condition
- **Exploratory Data Analysis (EDA)** — Feature distributions across regimes
- **Baseline Performance** — KPIs, ROC Curves, Confusion Matrices for Regime 1
- **Drift Monitoring** — Density plots and metric degradation on Regime 2
- **Continual Learning Results** — Performance recovery after CL update
- **Feature Importance** — SVM coefficients and Gini importance rankings

---

## 📈 Results & Performance

After applying the Continual Adaptation Protocol, models typically:

- ✅ **Restore** F1-score and ROC-AUC to baseline levels
- ✅ **Exceed** baseline performance in many configurations
- ✅ Show **higher AUC** on ROC curves post-adaptation
- ✅ Surface **interpretable features** — Age, BMI, Procedure Costs consistently rank among top predictors

> Results are fully reproducible through the Streamlit dashboard without rerunning training from scratch, thanks to `joblib` artifact persistence.

---

## 🔁 Continual Learning Strategy

Traditional ML pipelines suffer from **catastrophic forgetting** when updated — overwriting learned historical patterns with new data. This project mitigates that by:

1. **Incremental updates** via `partial_fit` for SVM and MLP: only weight adjustments, no full retraining
2. **Controlled merging** for Decision Trees: historical and current data combined, preserving distributional breadth
3. **Threshold recalibration** after each update to maintain optimal precision-recall balance

This makes the system suitable for **real-world clinical deployment** where new patient cohorts continuously arrive and model maintenance must be efficient.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.8+ |
| ML Framework | scikit-learn |
| Dashboard | Streamlit |
| Data Processing | pandas, NumPy |
| Visualization | matplotlib, seaborn |
| Model Persistence | joblib |
| Dataset | Synthea-MIMIC (synthetic EHR) |

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

Please ensure your code follows the existing module structure and includes relevant docstrings.

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Synthea](https://github.com/synthetichealth/synthea) for the synthetic patient generator
- [MIMIC](https://mimic.mit.edu/) for the clinical data schema inspiration
- [scikit-learn](https://scikit-learn.org/) for the ML ecosystem
- [Streamlit](https://streamlit.io/) for rapid dashboard development

---

<p align="center">
  Built with ❤️ for better, more adaptive clinical AI systems
</p>
