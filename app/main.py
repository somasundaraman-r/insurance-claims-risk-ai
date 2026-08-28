from pathlib import Path

import pandas as pd
import streamlit as st


# =========================================================
# PROJECT PATHS
# =========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_PATH = PROJECT_ROOT / "reports" / "claim_risk_scores.csv"
RAW_INSURANCE_PATH = PROJECT_ROOT / "data" / "insurance_data.csv"


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Insurance Claims Risk AI",
    page_icon="🛡️",
    layout="wide",
)


# =========================================================
# HEADER
# =========================================================

st.title("🛡️ Insurance Claims Risk AI")

st.markdown(
    """
    **AI-powered insurance claim risk assessment and fraud investigation dashboard**

    Analyze claim behavior, machine-learning anomaly signals,
    business risk indicators, and final risk classifications.
    """
)


# =========================================================
# DATA LOADING
# =========================================================

@st.cache_data
def load_risk_report():
    """Load the generated claim risk report."""

    if not REPORT_PATH.exists():
        return None

    return pd.read_csv(REPORT_PATH)


@st.cache_data
def load_raw_insurance_data():
    """Load raw insurance data when additional fields are required."""

    if not RAW_INSURANCE_PATH.exists():
        return None

    return pd.read_csv(RAW_INSURANCE_PATH)


df = load_risk_report()


# =========================================================
# REPORT CHECK
# =========================================================

if df is None:

    st.error("Risk report not found.")

    st.markdown(
        """
        Please run the risk scoring pipeline first:
        """
    )

    st.code(
        "python -m src.risk_scoring",
        language="powershell",
    )

    st.stop()


# =========================================================
# OPTIONAL RAW DATA ENRICHMENT
# =========================================================

raw_df = load_raw_insurance_data()

if raw_df is not None:

    enrichment_columns = [
        "TRANSACTION_ID",
        "CUSTOMER_ID",
        "POLICY_NUMBER",
        "CLAIM_STATUS",
        "CUSTOMER_NAME",
        "AGE",
        "TENURE",
        "MARITAL_STATUS",
        "EMPLOYMENT_STATUS",
        "NO_OF_FAMILY_MEMBERS",
        "RISK_SEGMENTATION",
        "HOUSE_TYPE",
        "SOCIAL_CLASS",
        "CUSTOMER_EDUCATION_LEVEL",
        "INCIDENT_SEVERITY",
        "AUTHORITY_CONTACTED",
        "ANY_INJURY",
        "POLICE_REPORT_AVAILABLE",
        "INCIDENT_STATE",
        "INCIDENT_CITY",
        "INCIDENT_HOUR_OF_THE_DAY",
        "AGENT_ID",
        "VENDOR_ID",
    ]

    available_enrichment_columns = [
        column
        for column in enrichment_columns
        if column in raw_df.columns
    ]

    if "TRANSACTION_ID" in df.columns:

        raw_subset = raw_df[
            available_enrichment_columns
        ].drop_duplicates(
            subset=["TRANSACTION_ID"]
        )

        columns_to_add = [
            column
            for column in available_enrichment_columns
            if column != "TRANSACTION_ID"
            and column not in df.columns
        ]

        if columns_to_add:

            df = df.merge(
                raw_subset[
                    ["TRANSACTION_ID"] + columns_to_add
                ],
                on="TRANSACTION_ID",
                how="left",
            )


# =========================================================
# REQUIRED COLUMN SAFETY
# =========================================================

# These columns are essential for the dashboard.
required_columns = [
    "TRANSACTION_ID",
    "INSURANCE_TYPE",
    "CLAIM_AMOUNT",
    "ANOMALY_SCORE",
    "BUSINESS_RISK_SCORE",
    "FINAL_RISK_SCORE",
    "RISK_LEVEL",
]


missing_required = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_required:

    st.error(
        "The risk report is missing required columns:"
    )

    st.write(missing_required)

    st.info(
        "Please regenerate the report using: "
        "`python -m src.risk_scoring`"
    )

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("🔎 Filters")


# ---------------------------------------------------------
# Insurance Type
# ---------------------------------------------------------

insurance_types = sorted(
    df["INSURANCE_TYPE"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

selected_insurance = st.sidebar.multiselect(
    "Insurance Type",
    options=insurance_types,
    default=insurance_types,
)


# ---------------------------------------------------------
# Risk Level
# ---------------------------------------------------------

risk_levels = sorted(
    df["RISK_LEVEL"]
    .dropna()
    .astype(str)
    .unique()
    .tolist()
)

selected_risk = st.sidebar.multiselect(
    "Risk Level",
    options=risk_levels,
    default=risk_levels,
)


# ---------------------------------------------------------
# Claim Status
# ---------------------------------------------------------

if "CLAIM_STATUS" in df.columns:

    claim_statuses = sorted(
        df["CLAIM_STATUS"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_status = st.sidebar.multiselect(
        "Claim Status",
        options=claim_statuses,
        default=claim_statuses,
    )

else:

    selected_status = None

    st.sidebar.info(
        "Claim Status is not available in the generated report."
    )


# =========================================================
# APPLY FILTERS
# =========================================================

filtered_df = df[
    df["INSURANCE_TYPE"].isin(selected_insurance)
    & df["RISK_LEVEL"].isin(selected_risk)
].copy()


if (
    selected_status is not None
    and "CLAIM_STATUS" in filtered_df.columns
):

    filtered_df = filtered_df[
        filtered_df["CLAIM_STATUS"].astype(str).isin(
            selected_status
        )
    ].copy()


# =========================================================
# NO RESULTS CHECK
# =========================================================

if filtered_df.empty:

    st.warning(
        "No claims match the selected filters."
    )

    st.stop()


# =========================================================
# RISK OVERVIEW
# =========================================================

st.subheader("📊 Risk Overview")


col1, col2, col3, col4 = st.columns(4)


# ---------------------------------------------------------
# Total Claims
# ---------------------------------------------------------

with col1:

    st.metric(
        "Total Claims",
        f"{len(filtered_df):,}",
    )


# ---------------------------------------------------------
# High Risk Claims
# ---------------------------------------------------------

with col2:

    high_count = (
        filtered_df["RISK_LEVEL"]
        .astype(str)
        .eq("HIGH")
        .sum()
    )

    st.metric(
        "High Risk Claims",
        f"{high_count:,}",
    )


# ---------------------------------------------------------
# Average Claim
# ---------------------------------------------------------

with col3:

    avg_claim = filtered_df["CLAIM_AMOUNT"].mean()

    st.metric(
        "Average Claim Amount",
        f"${avg_claim:,.0f}",
    )


# ---------------------------------------------------------
# Average Final Risk Score
# ---------------------------------------------------------

with col4:

    avg_score = filtered_df["FINAL_RISK_SCORE"].mean()

    st.metric(
        "Average Risk Score",
        f"{avg_score:.1f}",
    )


# =========================================================
# RISK DISTRIBUTION
# =========================================================

st.subheader("📈 Risk Distribution")


chart_col1, chart_col2 = st.columns(2)


# ---------------------------------------------------------
# Risk Level Distribution
# ---------------------------------------------------------

with chart_col1:

    risk_distribution = (
        filtered_df["RISK_LEVEL"]
        .value_counts()
        .rename_axis("Risk Level")
        .reset_index(name="Claims")
    )

    st.bar_chart(
        risk_distribution.set_index("Risk Level")
    )


# ---------------------------------------------------------
# Insurance Distribution
# ---------------------------------------------------------

with chart_col2:

    insurance_distribution = (
        filtered_df["INSURANCE_TYPE"]
        .value_counts()
        .rename_axis("Insurance Type")
        .reset_index(name="Claims")
    )

    st.bar_chart(
        insurance_distribution.set_index(
            "Insurance Type"
        )
    )


# =========================================================
# HIGHEST-RISK CLAIMS
# =========================================================

st.subheader("🚨 Highest-Risk Claims")


display_columns = [
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


available_columns = [
    column
    for column in display_columns
    if column in filtered_df.columns
]


top_claims = (
    filtered_df
    .sort_values(
        "FINAL_RISK_SCORE",
        ascending=False,
    )
    .head(20)
)


st.dataframe(
    top_claims[available_columns],
    use_container_width=True,
    hide_index=True,
)


# =========================================================
# CLAIM INVESTIGATION
# =========================================================

st.subheader("🔍 Claim Investigation")


search_id = st.text_input(
    "Enter Transaction ID or Customer ID",
    placeholder="Example: TXN00000671",
)


if search_id:

    transaction_match = (
        filtered_df["TRANSACTION_ID"]
        .astype(str)
        .str.contains(
            search_id,
            case=False,
            na=False,
        )
    )

    if "CUSTOMER_ID" in filtered_df.columns:

        customer_match = (
            filtered_df["CUSTOMER_ID"]
            .astype(str)
            .str.contains(
                search_id,
                case=False,
                na=False,
            )
        )

    else:

        customer_match = False


    search_results = filtered_df[
        transaction_match | customer_match
    ].copy()


    if search_results.empty:

        st.warning(
            "No matching claims found."
        )

    else:

        st.success(
            f"{len(search_results)} matching claim(s) found."
        )

        # =================================================
        # INVESTIGATION FOR SELECTED CLAIM
        # =================================================

        claim = search_results.iloc[0]

        transaction_id = claim.get(
            "TRANSACTION_ID",
            "Unknown",
        )

        st.markdown(
            f"### Claim: {transaction_id}"
        )


        # -------------------------------------------------
        # Claim KPI Cards
        # -------------------------------------------------

        inv_col1, inv_col2, inv_col3, inv_col4 = st.columns(4)


        with inv_col1:

            st.metric(
                "Final Risk Score",
                f"{claim.get('FINAL_RISK_SCORE', 0):.1f}",
            )


        with inv_col2:

            st.metric(
                "Risk Level",
                str(
                    claim.get(
                        "RISK_LEVEL",
                        "UNKNOWN",
                    )
                ),
            )


        with inv_col3:

            st.metric(
                "Claim Amount",
                f"${claim.get('CLAIM_AMOUNT', 0):,.0f}",
            )


        with inv_col4:

            st.metric(
                "Anomaly Score",
                f"{claim.get('ANOMALY_SCORE', 0):.1f}",
            )


        # =================================================
        # RISK SCORE COMPONENTS
        # =================================================

        st.markdown("### 📊 Risk Score Components")


        component_data = pd.DataFrame(
            {
                "Component": [
                    "Anomaly Score",
                    "Business Risk Score",
                    "Final Risk Score",
                ],
                "Score": [
                    float(
                        claim.get(
                            "ANOMALY_SCORE",
                            0,
                        )
                    ),
                    float(
                        claim.get(
                            "BUSINESS_RISK_SCORE",
                            0,
                        )
                    ),
                    float(
                        claim.get(
                            "FINAL_RISK_SCORE",
                            0,
                        )
                    ),
                ],
            }
        )


        st.bar_chart(
            component_data.set_index(
                "Component"
            )
        )


        # =================================================
        # WHY WAS THIS CLAIM FLAGGED?
        # =================================================

        st.markdown(
            "### 🚨 Why Was This Claim Flagged?"
        )


        reasons = []


        anomaly_score = float(
            claim.get(
                "ANOMALY_SCORE",
                0,
            )
        )


        business_score = float(
            claim.get(
                "BUSINESS_RISK_SCORE",
                0,
            )
        )


        final_score = float(
            claim.get(
                "FINAL_RISK_SCORE",
                0,
            )
        )


        claim_amount = float(
            claim.get(
                "CLAIM_AMOUNT",
                0,
            )
        )


        claim_ratio = claim.get(
            "CLAIM_TO_PREMIUM_RATIO",
            None,
        )


        reporting_delay = claim.get(
            "REPORTING_DELAY_DAYS",
            None,
        )


        # -------------------------------------------------
        # Anomaly signal
        # -------------------------------------------------

        if anomaly_score >= 75:

            reasons.append(
                f"Very high anomaly score ({anomaly_score:.1f}/100) "
                "indicates unusual claim behavior compared with "
                "the overall dataset."
            )

        elif anomaly_score >= 45:

            reasons.append(
                f"Elevated anomaly score ({anomaly_score:.1f}/100) "
                "indicates potentially unusual claim behavior."
            )


        # -------------------------------------------------
        # Claim-to-premium ratio
        # -------------------------------------------------

        if pd.notna(claim_ratio):

            if float(claim_ratio) >= 500:

                reasons.append(
                    f"Extremely high claim-to-premium ratio "
                    f"({float(claim_ratio):,.1f}x)."
                )

            elif float(claim_ratio) >= 200:

                reasons.append(
                    f"High claim-to-premium ratio "
                    f"({float(claim_ratio):,.1f}x)."
                )


        # -------------------------------------------------
        # Claim amount
        # -------------------------------------------------

        if claim_amount >= 75000:

            reasons.append(
                f"Large claim amount of "
                f"${claim_amount:,.0f} increases financial exposure."
            )


        # -------------------------------------------------
        # Reporting delay
        # -------------------------------------------------

        if pd.notna(reporting_delay):

            if float(reporting_delay) >= 5:

                reasons.append(
                    f"Claim was reported "
                    f"{float(reporting_delay):.0f} days after the loss event."
                )


        # -------------------------------------------------
        # Business risk
        # -------------------------------------------------

        if business_score >= 60:

            reasons.append(
                f"Business risk score is elevated "
                f"({business_score:.1f}/100)."
            )


        # -------------------------------------------------
        # Display reasons
        # -------------------------------------------------

        if reasons:

            for reason in reasons:

                st.markdown(
                    f"- {reason}"
                )

        else:

            st.info(
                "No individual high-severity risk signal "
                "was identified from the available indicators."
            )


        # =================================================
        # CLAIM DETAILS
        # =================================================

        st.markdown("### 📋 Claim Details")


        detail_columns = [
            "TRANSACTION_ID",
            "CUSTOMER_ID",
            "POLICY_NUMBER",
            "INSURANCE_TYPE",
            "CLAIM_STATUS",
            "CLAIM_AMOUNT",
            "PREMIUM_AMOUNT",
            "CLAIM_TO_PREMIUM_RATIO",
            "REPORTING_DELAY_DAYS",
            "INCIDENT_SEVERITY",
            "AUTHORITY_CONTACTED",
            "ANY_INJURY",
            "POLICE_REPORT_AVAILABLE",
            "INCIDENT_STATE",
            "INCIDENT_CITY",
            "INCIDENT_HOUR_OF_THE_DAY",
            "RISK_SEGMENTATION",
            "HOUSE_TYPE",
            "SOCIAL_CLASS",
            "CUSTOMER_EDUCATION_LEVEL",
            "AGE",
            "TENURE",
            "AGENT_ID",
            "VENDOR_ID",
        ]


        available_detail_columns = [
            column
            for column in detail_columns
            if column in search_results.columns
        ]


        detail_data = (
            claim[available_detail_columns]
            .rename_axis("Field")
            .reset_index(name="Value")
        )


        st.dataframe(
            detail_data,
            use_container_width=True,
            hide_index=True,
        )


        # =================================================
        # INVESTIGATION RECOMMENDATION
        # =================================================

        st.markdown(
            "### 👮 Investigation Recommendation"
        )


        if final_score >= 75:

            st.error(
                "HIGH RISK — Prioritize this claim for manual "
                "investigation and supporting-document verification."
            )

        elif final_score >= 45:

            st.warning(
                "MEDIUM RISK — Perform additional review before "
                "final claim processing."
            )

        else:

            st.success(
                "LOW RISK — No immediate high-priority "
                "investigation is indicated."
            )


# =========================================================
# RISK SCORE INTERPRETATION
# =========================================================

st.markdown("### ℹ️ Risk Score Interpretation")


st.markdown(
    """
**Final Risk Score**

- 🔴 **HIGH:** Score ≥ 75
- 🟠 **MEDIUM:** Score ≥ 45 and < 75
- 🟢 **LOW:** Score < 45

The final score combines machine-learning anomaly signals
with business risk indicators to prioritize claims for
further investigation.

**Important:** A high-risk classification is an investigation
priority signal, not a determination that fraud has occurred.
"""
)


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Insurance Claims Risk AI • Machine Learning + Explainable Risk Analytics"
)