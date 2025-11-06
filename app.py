import streamlit as st
import pickle
import numpy as np
import pandas as pd

# --- Configuration ---
# Ensure the model file is accessible in the same directory as this script.
MODEL_PATH = 'best_model_for_insurance.pkl'
# The R2 score (0.7202) is taken directly from the "Insurance project.ipynb" file.
R2_SCORE = 0.72 

@st.cache_resource
def load_model():
    """Loads the pre-trained machine learning model."""
    try:
        with open(MODEL_PATH, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"Error: Model file '{MODEL_PATH}' not found. Please ensure it is uploaded.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
lr_model = load_model()

# --- Streamlit App Layout ---
st.set_page_config(
    page_title="Enhanced Insurance Charges Predictor", 
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for aesthetics
st.markdown("""
    <style>
    .big-font {
        font-size:32px !important;
        font-weight: bold;
        color: #007bff;
        text-align: center;
        margin-bottom: 20px;
    }
    .result-box {
        padding: 25px;
        border-radius: 12px;
        border: 3px solid #1a7d1a;
        background-color: #e8f9e8;
        text-align: center;
        margin-top: 25px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    .stSlider > div > div > div:nth-child(2) {
        background-color: #007bff; /* Slider track color */
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">💰 Insurance Charges Calculator (LR Model)</p>', unsafe_allow_html=True)


# --- SIDEBAR: Model Insights ---
with st.sidebar:
    st.title("Model Insights")
    st.subheader("Linear Regression Details")
    
    if lr_model is not None and hasattr(lr_model, 'coef_') and hasattr(lr_model, 'intercept_'):
        # Display Coefficients
        # Features: age, bmi, smoker_yes (from the pickled model)
        features = ['Age', 'BMI', 'Smoker (Yes)']
        coefficients = lr_model.coef_
        coef_df = pd.DataFrame({
            'Feature': features,
            'Coefficient (Increase in Charges)': coefficients.round(2)
        })
        st.dataframe(coef_df, hide_index=True)
        
        st.caption("These values indicate how much each feature (when increased by one unit) contributes to the total predicted charge.")
        
        st.markdown(f"**Intercept:** ${lr_model.intercept_:.2f}")
        st.info(f"**Model R² Score (Test Data):** {R2_SCORE*100:.1f}%")

        st.markdown("""
            ---
            **Interpretation:**
            - **Smoker Status** has the largest impact, drastically increasing the predicted cost (Smoker is a binary feature (0 or 1), so the coefficient is the full cost increase).
            - **Age** is the next major factor ($/year).
            - **BMI** also positively correlates with charges ($/BMI unit).
        """)
    else:
        st.warning("Model details cannot be displayed until the model file is successfully loaded.")


# --- MAIN APP: User Inputs ---
st.subheader("Customer Profile")

if lr_model is not None:
    with st.form("prediction_form"):
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Input 1: Age (Updated to number input)
            age = st.number_input("1. Age", min_value=18, max_value=65, value=30, step=1, format="%d",
                                  help="Enter the customer's age (18-65).")
        
        with col2:
            # Input 2: BMI
            bmi = st.number_input("2. BMI (Body Mass Index)", min_value=15.0, max_value=50.0, value=25.0, step=0.1, format="%.2f",
                                  help="Enter BMI. Values > 30 are often classified as obese.")
            
        # Input 3: Smoker Status
        st.write("---")
        smoker_status = st.radio(
            "3. Smoker Status",
            ('No', 'Yes'),
            horizontal=True,
            key="smoker_radio"
        )
        
        # Prediction Button
        submitted = st.form_submit_button("Calculate Charges 🚀", type="primary")

        # --- Prediction Logic ---
        if submitted:
            # Convert categorical/user-friendly input to model feature format
            smoker_yes = 1 if smoker_status == 'Yes' else 0

            # Prepare the input data array [age, bmi, smoker_yes]
            input_data = np.array([[age, bmi, smoker_yes]])
            
            try:
                # Make prediction
                prediction = lr_model.predict(input_data)[0]
                
                # Format the prediction for display
                # Ensure charges are not negative (setting a minimum for safety)
                predicted_charges = max(500, prediction) 
                formatted_charges = f"${predicted_charges:,.2f}"
                
                # --- Display Results ---
                st.markdown(
                    f"""
                    <div class="result-box">
                        <p style="font-size:18px; font-weight:600;">The Estimated Annual Insurance Charge is:</p>
                        <p style="font-size:56px; font-weight:bolder; color:#1a7d1a;">{formatted_charges}</p>
                    </div>
                    """, 
                    unsafe_allow_html=True
                )
                
                # Dynamic Insight based on risk factors
                if smoker_yes == 1:
                    st.warning("🚨 **Smoker Warning:** The predicted charge is significantly higher due to the customer's smoker status.")
                elif bmi >= 30:
                    st.warning("🟡 **Health Note:** A BMI of 30 or higher is contributing to increased charges.")
                else:
                    st.success("👍 **Prediction Insight:** The customer profile suggests relatively lower risk factors (non-smoker, healthy BMI).")


            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")

# If model load failed
else:
    st.error("Application cannot run without the model. Please check the sidebar for file loading details.")

st.write("---")
st.caption("This prediction is based on a simple Linear Regression model using only Age, BMI, and Smoker status. For a more accurate quote, more features (like region, number of children, etc.) would be required.")
st.divider()
st.caption(''' Built with ❤️ using Streamlit and scikit-learn. This dashboard emphasizes clarity, speed, and reproducibility. ''')
