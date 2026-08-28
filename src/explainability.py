from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "anomaly_model.joblib"


def load_anomaly_model():
    """Load the trained anomaly detection pipeline."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def get_transformed_feature_names(pipeline):
    """Return feature names after preprocessing."""

    preprocessor = pipeline.named_steps["preprocessor"]

    feature_names = (
        preprocessor
        .get_feature_names_out()
        .tolist()
    )

    return feature_names


def explain_claim(pipeline, claim_df):
    """
    Generate SHAP feature contributions for one claim.

    The returned values represent the relative contribution
    of transformed features to the Isolation Forest model output.
    """

    preprocessor = pipeline.named_steps["preprocessor"]

    anomaly_model = pipeline.named_steps["model"]

    transformed = preprocessor.transform(
        claim_df
    )

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    transformed = np.asarray(
        transformed,
        dtype=float,
    )

    feature_names = get_transformed_feature_names(
        pipeline
    )

    explainer = shap.TreeExplainer(
        anomaly_model
    )

    shap_values = explainer.shap_values(
        transformed
    )

    shap_values = np.asarray(
        shap_values
    )

    if shap_values.ndim == 1:
        shap_values = shap_values.reshape(
            1,
            -1,
        )

    values = shap_values[0]

    result = pd.DataFrame(
        {
            "Feature": feature_names,
            "SHAP_VALUE": values,
            "ABS_SHAP_VALUE": np.abs(values),
        }
    )

    result = (
        result
        .sort_values(
            "ABS_SHAP_VALUE",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return result


def get_top_explanations(
    pipeline,
    claim_df,
    top_n=10,
):
    """Return the most influential features for a claim."""

    explanation = explain_claim(
        pipeline,
        claim_df,
    )

    return explanation.head(
        top_n
    )


def explain_claim_from_report(
    pipeline,
    full_df,
    transaction_id,
):
    """
    Locate a claim by transaction ID and
    generate its SHAP explanation.
    """

    if "TRANSACTION_ID" not in full_df.columns:
        raise ValueError(
            "TRANSACTION_ID column is required."
        )

    matches = full_df[
        full_df["TRANSACTION_ID"].astype(str)
        == str(transaction_id)
    ]

    if matches.empty:
        return None

    claim = matches.iloc[[0]].copy()

    return get_top_explanations(
        pipeline,
        claim,
    )