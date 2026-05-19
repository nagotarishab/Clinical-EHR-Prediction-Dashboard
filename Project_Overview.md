```
=============================================================================
CLINICAL EHR PREDICTION DASHBOARD - COMPREHENSIVE OVERVIEW
=============================================================================

1. PROJECT OVERVIEW
----------------------------------------------------------------------------This project implements a robust, automated machine learning pipeline and an
interactive Streamlit dashboard designed to predict clinical conditions (such
as Hypertension, Diabetes, etc.) using synthetic Electronic Health Records
(EHR).
A central focus of the project is handling "Temporal Data Drift"—the phenomenon
where clinical data distributions and relationships change over time, degrading
model performance.

To solve this, the pipeline splits data chronologically (Historical vs. Current
Cohorts) and applies "Continual Learning" (CL). Base models are trained on
historical data, evaluated on newer data to detect drift, and then continuously
updated using new data streams without training from scratch.

----------------------------------------------------------------------------2. ARCHITECTURE AND CODEBASE COMPONENTS
----------------------------------------------------------------------------The project is structured efficiently into distinct operational modules:

- `src/data_processing.py`: This script is the backbone of feature engineering.
It loads all raw data files, merges them on Patient, Encounter, and Claim
relationships, and creates a unified dataset with one row per Encounter.
It applies temporal splitting (e.g., pre/post 2018), imputes missing data
(Mean/Mode), standardizes numerical features, and One-Hot encodes categories.

- `src/models.py`: This file defines the `ModelPipeline` class. It manages
three algorithms: Decision Tree Classifier, Support Vector Machine (SGD),
and a Multi-Layer Perceptron (MLP) Neural Network. It handles:
* Baseline Training (on Historical Data).
* Threshold Tuning (Optimized for F1-score).
* Continual Learning (using partial_fit for SVM/MLP and refitting for Trees).
* Saving/Loading state (via joblib into the `artifacts/` folder).

- `TeamXX_Assignment2_dashboard.py`: A premium interactive Streamlit web
application.
It provides the user interface for pipeline configuration, data exploration
(EDA),
and monitoring. It renders performance KPIs, density drift graphs, confusion
matrices, and feature importance over the system's lifecycle.

----------------------------------------------------------------------------3. EXCEL SHEETS / CSV DATA DICTIONARY (synthea-mimic)
----------------------------------------------------------------------------The `synthea-mimic/csv/` directory contains standard healthcare domain data
tables
representing a simulated hospital environment. During processing, these are
aggregated to the "Encounter" (visit) level:

- **patients.csv**: Patient demographics including Birthdate, Race, Ethnicity,
Gender, and Income.
- **encounters.csv**: The central table. Each row represents a clinic visit or
hospital admission. Features are joined to this table using `Id` or `PATIENT`.
- **conditions.csv**: Contains diagnoses. Used to generate the binary `TARGET`
variable (e.g., 1 if the patient was diagnosed with "hypertension" during
the encounter, 0 otherwise).
- **observations.csv**: Captures clinical vitals and lab results (e.g., BMI,
Systolic/Diastolic Blood Pressure, Heart Rate). The code extracts the `mean`
and `variance` of these metrics per encounter.
- **medications, procedures, immunizations, supplies, careplans, devices,

```

```
imaging_studies.csv**: These tables log clinical interventions. They are
aggregated into counts (e.g., number of procedures per encounter) and cost
sums (e.g., total medication cost) to be used as predictive features.
- **organizations & providers.csv**: Information about the treating hospital
(Revenue, Utilization) and the individual practitioner (Specialty, Gender).
- **payer_transitions.csv**: Tracks changes in patient healthcare insurance.
- **claims & claims_transactions.csv**: Billing, financial, and insurance claim
data generated after the encounter. Includes outstanding amounts and total
cost.

----------------------------------------------------------------------------4. MACHINE LEARNING PIPELINE & RESULTS WORKFLOW
----------------------------------------------------------------------------Stage 1: Temporal Splitting & Drift Sensation:
The dataset is split using a chronological threshold. The "Historical" dataset
acts as Regime 1 (Train/Test), while the "Current" dataset acts as Regime 2.
You will observe drift—where prevalence rates of diseases and distributions of
numerical variables (like Age or Billing Costs) shift between the two regimes.

Stage 2: Baseline Model Training:
Decision Trees, SVMs, and Neural Networks are trained purely on Historical Data.
Because medical data is highly imbalanced, `class_weight='balanced'` is
utilized.
Metrics evaluated include ROC-AUC, F1-Score, Precision, Recall, and Accuracy.

Stage 3: Degradation Evaluation (Drift Impact):
When Baseline models attempt to predict the Current (Regime 2) dataset, their
performance (F1-score / ROC-AUC) typically drops compared to their baseline
due to the shifted data distributions. The dashboard visualizes this degradation
using side-by-side Confusion Matrices and bar charts.

Stage 4: Continual Learning Update (The Solution):
To fix the degradation without a complete retrain, "Continual Learning" is
triggered.
- For MLP and SVM models: Uses Stochastic Gradient Descent `partial_fit` to
incrementally adjust weights using the new data.
- For Decision Tree: Since trees cannot be easily incrementally updated in
standard
sklearn, it merges historical and current subsets to refit efficiently.
- Threshold Tuning: Prediction threshold probabilities are recalibrated to
optimize
classification boundaries.

Final Results Observation:
After executing the Continual Adaptation Protocol, the model restores and often
exceeds its predictive power on the Current dataset. The Streamlit dashboard
visually
confirms this through:
- Improved ROC Curves (Higher AUC).
- Increased F1 scores represented in green trend bars.
- Clear feature interpretations (Top attributes affecting predictions, like Age,
BMI variables, or Procedure Costs, visualized via SVM Coefficients and Gini
Importance).

```

