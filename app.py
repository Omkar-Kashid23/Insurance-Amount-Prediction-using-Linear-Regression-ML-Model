
# app.py
# Professional Streamlit app for Insurance Charge Prediction
# Author: Top 1–5% style 😉
# ----------------------------------------------------------
import os
import io
import json
import pickle
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

APP_TITLE = "Insurance Charge Prediction — Pro Dashboard"
MODEL_PATH = os.environ.get("MODEL_PATH", "best_model_for_insurance.pkl")

# ----------------------------
# Utilities
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_model(model_path: str) -> BaseEstimator:
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model

def _safe_getattr(obj, name: str, default=None):
    try:
        return getattr(obj, name, default)
    except Exception:
        return default

def infer_feature_meta(model: BaseEstimator) -> Tuple[List[str], List[str]]:
    \"\"\"Try to infer categorical & numeric feature names from a sklearn Pipeline or estimator.
    Returns (numeric_features, categorical_features).\"\"\"
    # Default heuristic for classic insurance dataset
    default_numeric = ["age", "bmi", "children"]
    default_categorical = ["sex", "smoker", "region"]

    # If it's a Pipeline with a ColumnTransformer, try to read feature names
    if isinstance(model, Pipeline):
        # Search steps for ColumnTransformer
        for name, step in model.steps:
            if isinstance(step, ColumnTransformer):
                numeric_features, categorical_features = [], []
                try:
                    for trans_name, trans, cols in step.transformers_:
                        if cols is None or cols == "drop":
                            continue
                        # Some transformers store columns as list/array/Index
                        if hasattr(cols, "tolist"):
                            use_cols = cols.tolist()
                        else:
                            use_cols = list(cols)
                        # Heuristic: if transformer has get_feature_names_out, assume numeric
                        # else use the given name to group
                        # But here we trust the ColumnTransformer setup: it usually names groups like 'num', 'cat'
                        if "num" in trans_name.lower():
                            numeric_features.extend(use_cols)
                        elif "cat" in trans_name.lower():
                            categorical_features.extend(use_cols)
                        else:
                            # Fallback: decide by dtype if available from model.feature_names_in_
                            pass
                except Exception:
                    pass

                # If we captured anything, return it
                if (locals().get("numeric_features") and len(numeric_features) > 0) or \
                   (locals().get("categorical_features") and len(categorical_features) > 0):
                    return (sorted(set(numeric_features)), sorted(set(categorical_features)))

        # If Pipeline had feature_names_in_
        fni = _safe_getattr(model, "feature_names_in_", None)
        if fni is not None:
            # Try to map common names
            cols = list(fni)
            cats = [c for c in cols if c.lower() in {"sex", "smoker", "region"}]
            nums = [c for c in cols if c.lower() in {"age", "bmi", "children"}]
            if cats or nums:
                return (nums or default_numeric, cats or default_categorical)

    # If plain estimator with feature_names_in_
    fni = _safe_getattr(model, "feature_names_in_", None)
    if fni is not None:
        cols = list(fni)
        cats = [c for c in cols if c.lower() in {"sex", "smoker", "region"}]
        nums = [c for c in cols if c.lower() in {"age", "bmi", "children"}]
        if cats or nums:
            return (nums or default_numeric, cats or default_categorical)

    return (default_numeric, default_categorical)

def validate_df(df: pd.DataFrame, expected_cols: List[str]) -> Tuple[bool, List[str]]:
    missing = [c for c in expected_cols if c not in df.columns]
    return (len(missing) == 0, missing)

def prettify_number(n: float) -> str:
    if n >= 1_00_00_000:  # 10M in Indian numbering is 1 crore
        return f"₹{n/1_00_00_000:.2f} Cr"
    if n >= 1_00_000:
        return f"₹{n/1_00_000:.2f} L"
    return f"₹{n:,.0f}"

# ----------------------------
# UI Helpers
# ----------------------------
def header():
    st.set_page_config(page_title=APP_TITLE, page_icon="💼", layout="wide")
    left, mid, right = st.columns([0.8, 0.1, 0.1])
    with left:
        st.title("💼 Insurance Charge Prediction")
        st.caption("Hybrid **AI x Business** dashboard — fast, robust, and production-ready.")
    with right:
        st.toggle("Dark mode", value=False, key="dark_mode", help="Toggles Streamlit theme in settings.")
    st.divider()

def sidebar_controls(cat_cols: List[str], num_cols: List[str]) -> Dict[str, Any]:
    st.sidebar.header("🔧 Prediction Controls")
    mode = st.sidebar.radio("Prediction mode", ["Single input", "Batch (CSV)"], horizontal=True)

    # Default options for classic dataset
    regions = ["southwest", "southeast", "northwest", "northeast"]
    sexes = ["male", "female"]
    yesno = ["yes", "no"]

    inputs = {}
    if mode == "Single input":
        st.sidebar.subheader("🧮 Input features")
        if "age" in num_cols:
            inputs["age"] = st.sidebar.number_input("Age", min_value=0, max_value=120, value=30, step=1)
        if "bmi" in num_cols:
            inputs["bmi"] = st.sidebar.number_input("BMI", min_value=0.0, max_value=100.0, value=27.5, step=0.1, format="%.1f")
        if "children" in num_cols:
            inputs["children"] = st.sidebar.number_input("Children", min_value=0, max_value=10, value=0, step=1)

        if "sex" in cat_cols:
            inputs["sex"] = st.sidebar.selectbox("Sex", options=sexes, index=0)
        if "smoker" in cat_cols:
            inputs["smoker"] = st.sidebar.selectbox("Smoker", options=yesno, index=1)
        if "region" in cat_cols:
            inputs["region"] = st.sidebar.selectbox("Region", options=regions, index=0)

        return {"mode": mode, "payload": inputs}

    else:
        st.sidebar.subheader("📥 Upload CSV")
        st.sidebar.caption("Include columns: age, bmi, children, sex, smoker, region (lowercase).")
        file = st.sidebar.file_uploader("Upload .csv", type=["csv"])
        df = None
        if file is not None:
            try:
                df = pd.read_csv(file)
            except Exception as e:
                st.sidebar.error(f"Failed to read CSV: {e}")
        return {"mode": mode, "payload": df}

def model_predict(model: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
    preds = model.predict(X)
    # Ensure 1D
    return np.array(preds).reshape(-1,)

def kpis_section(preds: np.ndarray):
    st.subheader("📊 KPI Summary")
    c1, c2, c3 = st.columns(3)
    if len(preds) == 0:
        with c1: st.metric("Predicted charge", "—")
        with c2: st.metric("Median (batch)", "—")
        with c3: st.metric("Std Dev (batch)", "—")
        return

    with c1:
        st.metric("Predicted charge", prettify_number(float(preds[-1])))
    with c2:
        st.metric("Median (batch)" if len(preds) > 1 else "Median", prettify_number(float(np.median(preds))))
    with c3:
        st.metric("Std Dev (batch)" if len(preds) > 1 else "Std Dev", prettify_number(float(np.std(preds))))

def explain_rules_row(row: Dict[str, Any]) -> List[str]:
    \"\"\"Lightweight, human-readable rule-of-thumb explainer (no SHAP needed).\"\"\"
    messages = []
    age = row.get("age", None)
    bmi = row.get("bmi", None)
    smoker = str(row.get("smoker", "")).lower()
    region = str(row.get("region", "")).lower()

    if age is not None:
        if age >= 50:
            messages.append("Higher charge likely due to age 50+.")
        elif age <= 25:
            messages.append("Lower baseline due to young age.")
    if bmi is not None:
        if bmi >= 30:
            messages.append("BMI ≥ 30 suggests elevated risk.")
        elif bmi <= 20:
            messages.append("Lean BMI may reduce risk.")
    if "yes" in smoker:
        messages.append("Smoking strongly increases premiums.")
    if region in {"southeast"}:
        messages.append("Southeast region tends to have higher charges in classic datasets.")
    return messages

def render_single_prediction(model: BaseEstimator, num_cols: List[str], cat_cols: List[str], payload: Dict[str, Any]):
    df = pd.DataFrame([payload])
    preds = model_predict(model, df)
    kpis_section(preds)

    st.subheader("🧾 Prediction Detail")
    st.write("Latest input:")
    st.dataframe(df, use_container_width=True)
    st.success(f"Estimated insurance charge: **{prettify_number(float(preds[0]))}**")

    with st.expander("Why this prediction? (rule-of-thumb explanation)"):
        bullets = explain_rules_row(payload)
        if bullets:
            st.markdown("\\n".join([f"- {b}" for b in bullets]))
        else:
            st.write("No strong rule-of-thumb signals detected.")

def render_batch_predictions(model: BaseEstimator, num_cols: List[str], cat_cols: List[str], df: pd.DataFrame):
    expected = num_cols + cat_cols
    ok, missing = validate_df(df, expected)
    if not ok:
        st.error(f"Missing required columns: {missing}")
        st.stop()

    preds = model_predict(model, df)
    out = df.copy()
    out["predicted_charge"] = preds

    kpis_section(preds)

    st.subheader("📄 Batch Results")
    st.dataframe(out, use_container_width=True)

    # Download
    csv = out.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download predictions (CSV)", data=csv, file_name="insurance_predictions.csv", mime="text/csv")

def sample_csv(num_cols: List[str], cat_cols: List[str]) -> bytes:
    example = pd.DataFrame([
        {"age": 22, "bmi": 19.8, "children": 0, "sex": "male", "smoker": "no", "region": "northwest"},
        {"age": 57, "bmi": 31.2, "children": 2, "sex": "female", "smoker": "yes", "region": "southeast"},
    ])
    return example.to_csv(index=False).encode("utf-8")

# ----------------------------
# Main
# ----------------------------
def main():
    header()

    with st.spinner("Loading model..."):
        model = load_model(MODEL_PATH)

    # Infer schema
    num_cols, cat_cols = infer_feature_meta(model)

    # Hero section
    left, right = st.columns([0.62, 0.38])
    with left:
        st.markdown(\"\"\"
        ### 🚀 Ready to predict insurance charges
        Use the sidebar to enter a single record or upload a CSV for batch scoring.
        This app auto-detects your model's expected features and handles preprocessing via your sklearn pipeline.
        \"\"\")
        st.caption(f\"Expected numeric: `{num_cols}`  •  categorical: `{cat_cols}`\")
    with right:
        st.download_button("📄 Download sample CSV", data=sample_csv(num_cols, cat_cols),
                           file_name="sample_insurance_input.csv", mime="text/csv")
        st.info("Put your trained model file at **best_model_for_insurance.pkl** or set `MODEL_PATH` env.")

    # Controls
    controls = sidebar_controls(cat_cols, num_cols)

    # Routes
    if controls[\"mode\"] == \"Single input\":
        render_single_prediction(model, num_cols, cat_cols, controls[\"payload\"])
    else:
        df = controls[\"payload\"]
        if df is not None:
            render_batch_predictions(model, num_cols, cat_cols, df)
        else:
            st.warning("Upload a CSV to run batch predictions.")

    st.divider()
    st.caption(\"\"\"
    Built with ❤️ using Streamlit and scikit-learn. This dashboard emphasizes clarity, speed, and reproducibility.
    \"\"\")


if __name__ == \"__main__\":
    main()
