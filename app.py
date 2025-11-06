import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_lottie import st_lottie
import requests

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="💼 Insurance AI Dashboard",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = os.environ.get("MODEL_PATH", "best_model_for_insurance.pkl")

# ---------------- UTILITIES ----------------
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def prettify(num: float) -> str:
    return f"₹{num:,.0f}"

def create_feature_df(age, bmi, children, sex, smoker, region):
    return pd.DataFrame([{
        "age": age,
        "bmi": bmi,
        "children": children,
        "charges": 0,
        "sex_male": 1 if sex == "male" else 0,
        "smoker_yes": 1 if smoker == "yes" else 0,
        "region_northwest": 1 if region == "northwest" else 0,
        "region_southeast": 1 if region == "southeast" else 0,
        "region_southwest": 1 if region == "southwest" else 0,
    }])

def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# ---------------- SIDEBAR ----------------
def sidebar_inputs():
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/857/857681.png", width=80)
        st.title("Insurance Inputs 🧾")
        st.markdown("### Configure policy holder details")
        age = st.slider("Age", 18, 100, 35)
        bmi = st.slider("BMI (Body Mass Index)", 10.0, 50.0, 26.5)
        children = st.number_input("Children", 0, 10, 1)
        sex = st.radio("Sex", ["male", "female"], horizontal=True)
        smoker = st.radio("Smoker", ["yes", "no"], horizontal=True)
        region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])
        st.markdown("---")
        st.info("Adjust inputs and click **Predict** to see results.")
        return age, bmi, children, sex, smoker, region

# ---------------- MAIN APP ----------------
def main():
    st.markdown("<h1 style='text-align: center;'>💼 Insurance Charge Prediction Dashboard</h1>", unsafe_allow_html=True)
    st.caption("AI-powered predictive system with business-grade visualization")
    
    with st.spinner("Loading intelligent model..."):
        model = load_model()

    # Lottie animation (for top visual appeal)
    col1, col2 = st.columns([0.65, 0.35])
    with col2:
        lottie = load_lottie("https://assets4.lottiefiles.com/packages/lf20_cu8bpv.json")
        if lottie:
            st_lottie(lottie, height=220, key="insurance")
    with col1:
        st.markdown("""
        ### 🤖 Intelligent Insurance Estimation
        Enter customer details on the left sidebar.  
        The model predicts the **expected insurance charge** using real-world health and demographic data.
        """)
        st.markdown("> Built with ⚡ Streamlit + scikit-learn | Designed for clarity and performance")

    st.divider()

    # Sidebar inputs
    age, bmi, children, sex, smoker, region = sidebar_inputs()
    df = create_feature_df(age, bmi, children, sex, smoker, region)

    # Predict button
    if st.button("🚀 Predict Insurance Charge", use_container_width=True):
        try:
            pred = model.predict(df)[0]
            st.success(f"💰 Estimated Insurance Charge: **{prettify(pred)}**")
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted Amount", prettify(pred))
            c2.metric("BMI", f"{bmi}")
            c3.metric("Age", f"{age}")
            style_metric_cards()
            
            st.markdown("### 🔍 Prediction Details")
            st.dataframe(df, use_container_width=True)
            
            st.markdown("### 📈 Insights")
            st.bar_chart(pd.DataFrame({
                "Value": [age, bmi, children],
            }, index=["Age", "BMI", "Children"]))

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")

    st.divider()
    st.caption(
        "Built with ❤️ by Omkar Kashid | Elite AI Engineering | Streamlit • ML • Docker"
    )

# ---------------- ENTRY ----------------
if __name__ == "__main__":
    main()
