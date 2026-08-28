from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.claim_processor import build_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"


# These are identifiers, not useful predictive features.
ID_COLUMNS = [
    "TRANSACTION_ID",
    "CUSTOMER_ID",
    "POLICY_NUMBER",
    "AGENT_ID",
    "VENDOR_ID",
]


# Categorical features suitable for the anomaly model.
CATEGORICAL_FEATURES = [
    "INSURANCE_TYPE",
    "MARITAL_STATUS",
    "EMPLOYMENT_STATUS",
    "RISK_SEGMENTATION",
    "HOUSE_TYPE",
    "SOCIAL_CLASS",
    "CUSTOMER_EDUCATION_LEVEL",
    "CLAIM_STATUS",
    "INCIDENT_SEVERITY",
    "AUTHORITY_CONTACTED",
    "INCIDENT_STATE",
    "AGENT_STATE",
    "VENDOR_STATE",
]


# Numeric features representing claim/customer/incident behavior.
NUMERIC_FEATURES = [
    "PREMIUM_AMOUNT",
    "CLAIM_AMOUNT",
    "AGE",
    "TENURE",
    "NO_OF_FAMILY_MEMBERS",
    "ANY_INJURY",
    "POLICE_REPORT_AVAILABLE",
    "INCIDENT_HOUR_OF_THE_DAY",
    "REPORTING_DELAY_DAYS",
    "POLICY_AGE_DAYS",
    "CLAIM_TO_PREMIUM_RATIO",
    "LOG_CLAIM_AMOUNT",
    "LOG_PREMIUM_AMOUNT",
    "IS_NIGHT_INCIDENT",
    "MISSING_AUTHORITY_CONTACT",
    "MISSING_VENDOR",
    "MISSING_EDUCATION",
    "AGENT_CLAIM_COUNT",
    "AGENT_AVG_CLAIM_AMOUNT",
    "VENDOR_CLAIM_COUNT",
    "VENDOR_AVG_CLAIM_AMOUNT",
    "AGENT_TENURE_DAYS",
]


def build_anomaly_pipeline():
    """Create preprocessing + Isolation Forest pipeline."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )

    anomaly_model = IsolationForest(
        n_estimators=300,
        contamination=0.05,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", anomaly_model),
        ]
    )

    return pipeline


def train_anomaly_model(df):
    """Train the anomaly detection model."""

    features = df[
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    ].copy()

    pipeline = build_anomaly_pipeline()

    pipeline.fit(features)

    return pipeline


def calculate_anomaly_scores(pipeline, df):
    """Generate normalized anomaly scores from 0 to 100."""

    features = df[
        NUMERIC_FEATURES + CATEGORICAL_FEATURES
    ].copy()

    # Isolation Forest:
    # larger decision_function values = more normal.
    raw_scores = pipeline.decision_function(features)

    # Reverse direction so larger values indicate greater risk.
    risk_values = -raw_scores

    # Normalize to 0-100.
    min_score = risk_values.min()
    max_score = risk_values.max()

    if max_score == min_score:
        normalized_scores = np.zeros(len(risk_values))
    else:
        normalized_scores = (
            (risk_values - min_score)
            / (max_score - min_score)
            * 100
        )

    return normalized_scores


def assign_risk_level(score):
    """Convert numerical risk score into business-friendly risk levels."""

    if score >= 75:
        return "HIGH"
    elif score >= 45:
        return "MEDIUM"
    else:
        return "LOW"


def train_and_score():
    """Train the anomaly model and score all claims."""

    print("Loading and processing data...")

    df = build_features()

    print(f"Dataset shape: {df.shape}")

    print("\nTraining anomaly detection model...")

    pipeline = train_anomaly_model(df)

    scores = calculate_anomaly_scores(
        pipeline,
        df,
    )

    df["ANOMALY_SCORE"] = scores
    df["RISK_LEVEL"] = df["ANOMALY_SCORE"].apply(
        assign_risk_level
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = MODEL_DIR / "anomaly_model.joblib"

    joblib.dump(
        pipeline,
        model_path,
    )

    output_columns = [
        "TRANSACTION_ID",
        "CUSTOMER_ID",
        "INSURANCE_TYPE",
        "CLAIM_AMOUNT",
        "CLAIM_STATUS",
        "INCIDENT_SEVERITY",
        "ANOMALY_SCORE",
        "RISK_LEVEL",
    ]

    scored_claims = df[output_columns].copy()

    print("\nRisk distribution:")
    print(
        scored_claims["RISK_LEVEL"]
        .value_counts()
        .sort_index()
    )

    print("\nTop 10 highest-risk claims:")

    print(
        scored_claims
        .sort_values(
            "ANOMALY_SCORE",
            ascending=False,
        )
        .head(10)
        .to_string(index=False)
    )

    print(f"\nModel saved to: {model_path}")

    return pipeline, scored_claims


if __name__ == "__main__":
    train_and_score()