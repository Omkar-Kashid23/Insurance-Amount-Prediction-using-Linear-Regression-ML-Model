import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st

# ---------------- CONFIG ----------------
st.set_page_config(page_title="💼 Insurance Charge Prediction", page_icon="💰", layout="wide")

MODEL_PATH = os.environ.get("MODEL_PATH", "best_model_for_insurance.pkl")

# ---------------- UTILITIES ----------------
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def prettify(num: float) -> str:
    return f"₹{num:,.0f}"

def create_feature_df(age, bmi, children, sex, smoker, region):
    """
    Build a DataFrame with the exact feature names expected by the model.
    """
    features = {
        "age": age,
        "bmi": bmi,
        "children": children,
        "charges": 0,  # dummy placeholder for compatibility
        "sex_male": 1 if sex == "male" else 0,
        "smoker_yes": 1 if smoker == "yes" else 0,
        "region_northwest": 1 if region == "northwest" else 0,
        "region_southeast": 1 if region == "southeast" else 0,
        "region_southwest": 1 if region == "southwest" else 0,
    }
    return pd.DataFrame([features])

# ---------------- UI COMPONENTS ----------------
def sidebar_inputs():
    st.sidebar.header("🔧 Input Features")
    age = st.sidebar.number_input("Age", 18, 100, 30)
    bmi = st.sidebar.number_input("BMI", 10.0, 60.0, 25.0)
    children = st.sidebar.number_input("Children", 0, 10, 0)
    sex = st.sidebar.selectbox("Sex", ["male", "female"])
    smoker = st.sidebar.selectbox("Smoker", ["yes", "no"])
    region = st.sidebar.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])
    return age, bmi, children, sex, smoker, region

# ---------------- MAIN APP ----------------
def main():
    st.title("💼 Insurance Charge Prediction — AI × Business Dashboard")
    st.caption("Predict insurance charges using a trained ML model")

    with st.spinner("Loading model..."):
        model = load_model()

    st.success("Model loaded successfully ✅")

    # Sidebar inputs
    age, bmi, children, sex, smoker, region = sidebar_inputs()

    if st.button("🔮 Predict Insurance Charge"):
        df = create_feature_df(age, bmi, children, sex, smoker, region)
        st.write("**Model input:**")
        st.dataframe(df, use_container_width=True)

        try:
            prediction = model.predict(df)[0]
            st.success(f"💰 Estimated Insurance Charge: {prettify(prediction)}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

    st.divider()
    st.caption("Built with ❤️ using Streamlit and scikit-learn — by Omkar Kashid")

# ---------------- ENTRY ----------------
if __name__ == "__main__":
    main()
