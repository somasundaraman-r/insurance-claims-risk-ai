# 🛡️ Insurance Claims Risk AI

AI-powered insurance claims risk assessment and fraud investigation system built using Python, Machine Learning, and Streamlit.

The project analyzes insurance claims using **anomaly detection**, **business risk signals**, and **explainable AI** to identify claims that may require further investigation.

---

## 🚀 Project Overview

Insurance companies process a large number of claims, making it difficult to manually identify unusual or potentially fraudulent claims.

This project provides an automated risk-scoring pipeline that:

- Processes insurance claim data
- Engineers relevant risk features
- Detects anomalous claims using Machine Learning
- Calculates business risk indicators
- Generates a final risk score
- Classifies claims into LOW, MEDIUM, and HIGH risk
- Provides explainability for individual claims
- Displays results through an interactive Streamlit dashboard

---

## 🧠 Key Features

### 1. Claim Processing

The claim processing pipeline combines insurance, employee, and vendor information and creates features required for risk analysis.

### 2. Anomaly Detection

Machine Learning is used to identify claims that behave differently from the overall claim population.

Anomaly scores are normalized to a 0–100 scale.

### 3. Business Risk Scoring

Additional business rules are used to identify potentially suspicious claim characteristics, including:

- Claim amount
- Premium amount
- Claim-to-premium ratio
- Reporting delay
- Other claim-related risk indicators

### 4. Final Risk Score

The system combines:

- Machine Learning anomaly score
- Business risk score

to produce a final risk score.

Claims are then classified as:

| Risk Level | Score |
|---|---:|
| 🔴 HIGH | ≥ 75 |
| 🟠 MEDIUM | ≥ 45 and < 75 |
| 🟢 LOW | < 45 |

### 5. Explainable AI

The project uses SHAP to provide feature-level explanations for individual claim risk predictions.

This helps investigators understand which features contributed most to the model's assessment.

### 6. Interactive Dashboard

The Streamlit dashboard provides:

- Risk overview
- Total claim count
- High-risk claim count
- Average claim amount
- Average risk score
- Risk distribution
- Insurance-type distribution
- Highest-risk claims
- Claim investigation search
- Individual claim risk details
- Risk score components
- Investigation recommendations
- Explainable AI insights

---

## 🏗️ Project Structure

```text
insurance-claims-risk-ai/
│
├── app/
│   └── main.py
│
├── src/
│   ├── __init__.py
│   ├── claim_processor.py
│   ├── fraud_detection.py
│   ├── risk_scoring.py
│   └── explainability.py
│
├── tests/
│   └── test_claims.py
│
├── data/
│   └── *.csv
│
├── models/
│   └── generated model files
│
├── reports/
│   └── generated risk reports
│
├── .gitignore
├── requirements.txt
├── main.py
└── README.md


# Setup & Run Instructions

## 1. Clone the Repository

```powershell
git clone https://github.com/somasundaraman-r/insurance-claims-risk-ai.git

## 2. Enter the project folder

cd insurance-claims-risk-ai

## 3. Create Virtual Environment

python -m venv .venv

## 4. Activate Virtual Environment

.venv\Scripts\Activate.ps1

## 5. Install Dependencies

pip install -r requirements.txt

## 6. Add dataset

data\insurance_data.csv

## 7. Generate Risk scores

python -m src.risk_scoring

## 8. Start the Streamlit dashboard

python -m streamlit run app/main.py

## 9. Open the dashboard

http://localhost:8501
