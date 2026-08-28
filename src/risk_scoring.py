from pathlib import Path

import joblib
import pandas as pd

from src.claim_processor import build_features
from src.fraud_detection import (
    calculate_anomaly_scores,
    train_anomaly_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"


def calculate_business_risk(df):
    """
    Calculate additional rule-based business risk signals.

    The goal is not to replace the ML anomaly model.
    These signals provide an interpretable business layer.
    """

    risk = pd.Series(0.0, index=df.index)

    # Extremely high claim-to-premium ratios can indicate
    # unusual financial exposure.
    ratio = df["CLAIM_TO_PREMIUM_RATIO"]

    risk += ratio.ge(500).astype(float) * 20
    risk += ratio.ge(200).astype(float) * 10

    # Large claims receive additional business risk weight.
    claim_amount = df["CLAIM_AMOUNT"]

    risk += claim_amount.ge(75000).astype(float) * 15
    risk += claim_amount.ge(50000).astype(float) * 10

    # Reporting delays may indicate unusual claim behavior.
    reporting_delay = df["REPORTING_DELAY_DAYS"]

    risk += reporting_delay.ge(7).astype(float) * 5
    risk += reporting_delay.ge(14).astype(float) * 5

    # Missing information reduces confidence in the claim
    # and therefore increases review priority.
    risk += df["MISSING_AUTHORITY_CONTACT"].astype(float) * 5
    risk += df["MISSING_VENDOR"].astype(float) * 5
    risk += df["MISSING_EDUCATION"].astype(float) * 2

    # Night-time incidents receive a small review signal.
    risk += df["IS_NIGHT_INCIDENT"].astype(float) * 3

    return risk.clip(upper=100)


def calculate_final_risk_score(anomaly_score, business_risk):
    """
    Combine ML anomaly risk with interpretable business risk.

    ML anomaly detection receives the larger weight because
    it captures multivariate unusual behavior.
    """

    final_score = (
        anomaly_score * 0.70
        + business_risk * 0.30
    )

    return final_score.clip(0, 100)


def assign_risk_level(score):
    """Convert final risk score into business risk levels."""

    if score >= 75:
        return "HIGH"

    if score >= 45:
        return "MEDIUM"

    return "LOW"


def generate_risk_scores():
    """Generate final claim risk scores."""

    print("Loading and processing data...")

    df = build_features()

    print(f"Dataset shape: {df.shape}")

    print("\nTraining anomaly detection model...")

    pipeline = train_anomaly_model(df)

    print("Calculating anomaly scores...")

    anomaly_scores = calculate_anomaly_scores(
        pipeline,
        df,
    )

    print("Calculating business risk signals...")

    business_risk = calculate_business_risk(df)

    final_scores = calculate_final_risk_score(
        anomaly_scores,
        business_risk,
    )

    df["ANOMALY_SCORE"] = anomaly_scores
    df["BUSINESS_RISK_SCORE"] = business_risk
    df["FINAL_RISK_SCORE"] = final_scores

    df["RISK_LEVEL"] = df["FINAL_RISK_SCORE"].apply(
        assign_risk_level
    )

    output_columns = [
        "TRANSACTION_ID",
        "CUSTOMER_ID",
        "INSURANCE_TYPE",
        "CLAIM_AMOUNT",
        "PREMIUM_AMOUNT",
        "CLAIM_TO_PREMIUM_RATIO",
        "REPORTING_DELAY_DAYS",
        "ANOMALY_SCORE",
        "BUSINESS_RISK_SCORE",
        "FINAL_RISK_SCORE",
        "RISK_LEVEL",
    ]

    scored_claims = df[output_columns].copy()

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = REPORT_DIR / "claim_risk_scores.csv"

    scored_claims.to_csv(
        output_path,
        index=False,
    )

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
            "FINAL_RISK_SCORE",
            ascending=False,
        )
        .head(10)
        .to_string(index=False)
    )

    print(f"\nRisk report saved to: {output_path}")

    return scored_claims


if __name__ == "__main__":
    generate_risk_scores()