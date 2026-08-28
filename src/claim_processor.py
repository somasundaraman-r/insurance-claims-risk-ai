from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def load_data():
    """Load the three source datasets."""

    insurance = pd.read_csv(DATA_DIR / "insurance_data.csv")
    employees = pd.read_csv(DATA_DIR / "employee_data.csv")
    vendors = pd.read_csv(DATA_DIR / "vendor_data.csv")

    return insurance, employees, vendors


def build_features():
    """Load, join, clean and engineer claim-level features."""

    insurance, employees, vendors = load_data()

    df = insurance.copy()

    # ---------------------------------------------------------
    # Date conversion
    # ---------------------------------------------------------
    date_columns = [
        "TXN_DATE_TIME",
        "POLICY_EFF_DT",
        "LOSS_DT",
        "REPORT_DT",
    ]

    for column in date_columns:
        df[column] = pd.to_datetime(df[column], errors="coerce")

    # ---------------------------------------------------------
    # Date-based features
    # ---------------------------------------------------------
    df["REPORTING_DELAY_DAYS"] = (
        df["REPORT_DT"] - df["LOSS_DT"]
    ).dt.total_seconds() / 86400

    df["POLICY_AGE_DAYS"] = (
        df["LOSS_DT"] - df["POLICY_EFF_DT"]
    ).dt.total_seconds() / 86400

    # ---------------------------------------------------------
    # Financial features
    # ---------------------------------------------------------
    df["CLAIM_TO_PREMIUM_RATIO"] = (
        df["CLAIM_AMOUNT"]
        / df["PREMIUM_AMOUNT"].replace(0, np.nan)
    )

    df["LOG_CLAIM_AMOUNT"] = np.log1p(df["CLAIM_AMOUNT"])

    df["LOG_PREMIUM_AMOUNT"] = np.log1p(df["PREMIUM_AMOUNT"])

    # ---------------------------------------------------------
    # Incident features
    # ---------------------------------------------------------
    df["IS_NIGHT_INCIDENT"] = (
        (df["INCIDENT_HOUR_OF_THE_DAY"] < 6)
        | (df["INCIDENT_HOUR_OF_THE_DAY"] >= 22)
    ).astype(int)

    df["MISSING_AUTHORITY_CONTACT"] = (
        df["AUTHORITY_CONTACTED"].isna()
    ).astype(int)

    df["MISSING_VENDOR"] = (
        df["VENDOR_ID"].isna()
    ).astype(int)

    df["MISSING_EDUCATION"] = (
        df["CUSTOMER_EDUCATION_LEVEL"].isna()
    ).astype(int)

    # ---------------------------------------------------------
    # Agent statistics
    # ---------------------------------------------------------
    agent_claim_counts = (
        df["AGENT_ID"]
        .value_counts()
        .rename("AGENT_CLAIM_COUNT")
    )

    df = df.join(agent_claim_counts, on="AGENT_ID")

    agent_avg_claim = (
        df.groupby("AGENT_ID")["CLAIM_AMOUNT"]
        .mean()
        .rename("AGENT_AVG_CLAIM_AMOUNT")
    )

    df = df.join(agent_avg_claim, on="AGENT_ID")

    # ---------------------------------------------------------
    # Vendor statistics
    # ---------------------------------------------------------
    vendor_claim_counts = (
        df["VENDOR_ID"]
        .value_counts()
        .rename("VENDOR_CLAIM_COUNT")
    )

    df = df.join(vendor_claim_counts, on="VENDOR_ID")

    vendor_avg_claim = (
        df.groupby("VENDOR_ID")["CLAIM_AMOUNT"]
        .mean()
        .rename("VENDOR_AVG_CLAIM_AMOUNT")
    )

    df = df.join(vendor_avg_claim, on="VENDOR_ID")

    # ---------------------------------------------------------
    # Join employee information
    # ---------------------------------------------------------
    employee_columns = [
        "AGENT_ID",
        "DATE_OF_JOINING",
        "CITY",
        "STATE",
    ]

    employees_subset = employees[employee_columns].copy()

    employees_subset = employees_subset.rename(
        columns={
            "CITY": "AGENT_CITY",
            "STATE": "AGENT_STATE",
        }
    )

    df = df.merge(
        employees_subset,
        on="AGENT_ID",
        how="left",
    )

    # ---------------------------------------------------------
    # Join vendor information
    # ---------------------------------------------------------
    vendor_columns = [
        "VENDOR_ID",
        "CITY",
        "STATE",
    ]

    vendors_subset = vendors[vendor_columns].copy()

    vendors_subset = vendors_subset.rename(
        columns={
            "CITY": "VENDOR_CITY",
            "STATE": "VENDOR_STATE",
        }
    )

    df = df.merge(
        vendors_subset,
        on="VENDOR_ID",
        how="left",
    )

    # ---------------------------------------------------------
    # Agent tenure at time of claim
    # ---------------------------------------------------------
    df["DATE_OF_JOINING"] = pd.to_datetime(
        df["DATE_OF_JOINING"],
        errors="coerce",
    )

    df["AGENT_TENURE_DAYS"] = (
        df["LOSS_DT"] - df["DATE_OF_JOINING"]
    ).dt.total_seconds() / 86400

    # ---------------------------------------------------------
    # Remove sensitive identifiers
    # ---------------------------------------------------------
    sensitive_columns = [
        "SSN",
        "ACCT_NUMBER",
        "ROUTING_NUMBER",
        "CUSTOMER_NAME",
        "ADDRESS_LINE1",
        "ADDRESS_LINE2",
        "EMP_ACCT_NUMBER",
        "EMP_ROUTING_NUMBER",
        "AGENT_NAME",
        "VENDOR_NAME",
    ]

    df = df.drop(
        columns=sensitive_columns,
        errors="ignore",
    )

    # ---------------------------------------------------------
    # Remove raw date columns after feature extraction
    # ---------------------------------------------------------
    df = df.drop(
        columns=[
            "TXN_DATE_TIME",
            "POLICY_EFF_DT",
            "LOSS_DT",
            "REPORT_DT",
            "DATE_OF_JOINING",
        ],
        errors="ignore",
    )

    return df


if __name__ == "__main__":
    data = build_features()

    print("Processed dataset shape:", data.shape)
    print("\nColumns:")
    print(data.columns.tolist())

    print("\nMissing values:")
    print(data.isna().sum().sort_values(ascending=False).head(15))