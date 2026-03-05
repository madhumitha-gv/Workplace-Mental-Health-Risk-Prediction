# Workplace Mental Health Risk Prediction

> **EY Hackathon — Predictive Model for Mental Health Risk**
> Built using traditional ML techniques to assess mental health risk in tech industry workplaces, with early detection and actionable insights for HR and mental health professionals.

---

## Overview

Mental health disorders are a growing concern in the workplace, yet traditional detection methods (static questionnaires, clinical interviews) are slow and often miss early warning signs. This project builds a data-driven predictive system trained on an anonymous survey of tech industry professionals to:

- Predict whether an employee currently has a mental health disorder
- Predict likelihood of seeking professional treatment
- Profile employees into distinct risk clusters
- Provide explainable, actionable insights for non-technical audiences

---

## Dataset

**Source:** Anonymous survey of tech industry professionals (`data/raw/mental_health.csv`)
**Size:** 1,467 responses × 63 features
**Covers:**
- Mental health status (current, past, diagnosed, treated)
- Workplace benefits, policies, and support resources
- Workplace culture, stigma, and openness
- Impact on productivity and work performance
- Demographic and job-related information

---

## What's Been Built

### 1. Data Preparation & Quality Assurance (`preprocessing.py`)
- Cleans and normalizes all categorical survey responses
- Handles missing/ambiguous values (`"I don't know"`, `"N/A"`, `"Not sure"`) with ordinal mappings
- Encodes ordinal variables (e.g., `"Very easy" → 4`, `"Never" → 0`) and applies label encoding for remaining categoricals
- Imputes remaining nulls with column medians

### 2. Feature Engineering — Three Composite Indices (`preprocessing.py`)

| Index | What It Measures | Key Input Features |
|---|---|---|
| **Mental Health Support Index** | Employer-provided benefits, resources, anonymity, formal communication | Benefits, care options, wellness program, leave ease, formal MH discussion |
| **Workplace Stigma Index** | Fear of negative consequences, observed discrimination | Employer disclosure fears, coworker/supervisor comfort (inverted), career impact, peer perception |
| **Organizational Openness Score** | Comfort discussing MH with managers and peers | Employer parity with physical health, anonymity protection, medical coverage, interview willingness |

All three indices are normalized to [0, 1] and saved into the feature dataframe for modeling and clustering.

### 3. Predictive Models (`app.py`, `models/`)

Two separate models trained on the top 10 most correlated features per target:

**Model 1 — Current Mental Health Disorder**
- Target: `"Do you currently have a mental health disorder?"`
- Algorithm: XGBoost Classifier
- Accuracy: ~82% | F1 Score: ~0.79

**Model 2 — Likelihood of Seeking Treatment**
- Target: `"Have you ever sought treatment for a mental health issue from a mental health professional?"`
- Algorithm: Gradient Boosting Classifier
- Accuracy: ~85% | F1 Score: ~0.83

Both models output a risk level (LOW / MODERATE / HIGH), probability score, and SHAP-based feature importance.

### 4. Explainability (SHAP)
- SHAP waterfall plots show which features drove each individual prediction
- Feature importance charts for non-technical audiences
- Composite index scores give a high-level wellness snapshot

### 5. Streamlit Web App (`app.py`)
Interactive UI for employees and HR teams:
- 15-question survey sidebar
- Real-time risk prediction with color-coded gauges
- SHAP explanation tab
- Personalised recommendations (prioritised by HIGH / MEDIUM / LOW)
- Analytics tab with benchmarks

### 6. FastAPI REST Backend (`api.py`)
Production-ready REST API:
- `POST /predict` — single employee prediction
- `POST /batch-predict` — batch processing with summary statistics
- `GET /health` — health check
- `GET /metrics` — model performance metrics
- `POST /train` — retrain with new CSV upload
- Auto-generated docs at `/docs`

---
---

## Quickstart

```bash
# 1. Clone the repo
git clone https://github.com/madhumitha-gv/Workplace-Mental-Health-Risk-Prediction.git
cd Workplace-Mental-Health-Risk-Prediction

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements_full.txt

# 4. Run the Streamlit app
streamlit run app/streamlit_app.py
# Opens at http://localhost:8501

# 5. (Optional) Run the API
uvicorn api.main:app --reload --port 8000
# Docs at http://localhost:8000/docs
```

---

## File Structure

```
Workplace-Mental-Health-Risk-Prediction/
│
├── data/
│   ├── raw/
│   │   └── mental_health.csv              # Source dataset (1,467 responses × 63 features)
│   ├── processed/                         # Cleaned & encoded outputs
│   └── outputs/                           # Predictions, cluster labels
│
├── notebooks/
│   └── Mental_Health_Analysis.ipynb       # Full reproducible analysis (WIP)
│
├── src/                                   # Core ML package
│   ├── data/
│   │   └── preprocessing.py               # Cleaning, encoding, composite index engineering
│   ├── features/
│   │   └── feature_engineering.py         # Correlation analysis + heatmaps
│   ├── models/
│   │   ├── train.py                       # Training pipeline (XGBoost + GradientBoosting)
│   │   └── evaluate.py                    # Metrics, SHAP explanations
│   └── clustering/
│       └── profiling.py                   # K-Means worker profiling (3 clusters)
│
├── models/                                # Saved model artifacts
│   ├── model_disorder.pkl                 # XGBoost — disorder prediction
│   ├── model_treatment.pkl                # GradientBoosting — treatment likelihood
│   ├── scaler.pkl                         # StandardScaler
│   └── feature_names.pkl                  # Feature list used at training time
│
├── app/
│   └── streamlit_app.py                   # Streamlit web application
│
├── api/
│   └── main.py                            # FastAPI REST API
│
├── tests/
│   └── test_api.py                        # API endpoint tests
│
├── reports/
│   ├── figures/                           # Heatmaps, SHAP plots, cluster charts
│   └── results_document.pdf              # 8-page hackathon results doc (WIP)
│
├── submissions/
│   └── submission_output.xlsx            # EY Excel submission template (WIP)
│
├── .gitignore
├── requirements_full.txt                  # All Python dependencies
├── DEPLOYMENT.md                          # Cloud deployment guide
├── STREAMLIT_API_GUIDE.md                 # App & API usage guide
└── README.md
```

---



## Tech Stack

| Component | Technology |
|---|---|
| Data Processing | pandas, numpy, scikit-learn |
| ML Models | XGBoost, Gradient Boosting |
| Explainability | SHAP |
| Web App | Streamlit |
| REST API | FastAPI + Uvicorn |
| Visualization | Plotly, Matplotlib, Seaborn |

---

## Team

Madhumitha Gannavaram — [GitHub](https://github.com/madhumitha-gv) | vgannava@iu.edu
