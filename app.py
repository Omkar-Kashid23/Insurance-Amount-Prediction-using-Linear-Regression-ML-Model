import streamlit as st
import pickle
import numpy as np

# --- Configuration ---
# Ensure the model file is accessible in the same directory as this script.
MODEL_PATH = 'best_model_for_insurance.pkl'

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
st.set_page_config(page_title="Insurance Charges Predictor", layout="centered")

st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
        color: #3b5998;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #00b300;
        background-color: #e6ffe6;
        text-align: center;
        margin-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-font">🏥 US Medical Insurance Charges Predictor</p>', unsafe_allow_html=True)
st.write("---")

if lr_model is not None:
    # --- User Inputs ---
    
    # Input 1: Age
    age = st.slider("1. Age", min_value=18, max_value=65, value=30, step=1)
    
    # Input 2: BMI
    # Based on the notebook, BMI ranges were roughly 15.96 to 47.29 after cleaning
    bmi = st.number_input("2. BMI (Body Mass Index)", min_value=15.0, max_value=55.0, value=25.0, step=0.1, format="%.2f")
    
    # Input 3: Smoker Status (One-Hot Encoded feature: smoker_yes)
    smoker_status = st.radio(
        "3. Smoker Status",
        ('No', 'Yes'),
        horizontal=True
    )
    
    # Convert categorical/user-friendly input to model feature format
    smoker_yes = 1 if smoker_status == 'Yes' else 0

    # --- Prediction Logic ---
    if st.button("Predict Charges 🔮", type="primary"):
        # Prepare the input data array [age, bmi, smoker_yes]
        # The model was trained on these three features, as confirmed by the pickle file.
        input_data = np.array([[age, bmi, smoker_yes]])
        
        try:
            # Make prediction
            prediction = lr_model.predict(input_data)[0]
            
            # Format the prediction for display
            # Ensure charges are not negative (though the LR model might predict it for edge cases)
            predicted_charges = max(0, prediction)
            formatted_charges = f"${predicted_charges:,.2f}"
            
            # --- Display Results ---
            st.markdown(
                f"""
                <div class="result-box">
                    <p style="font-size:16px;">The Estimated Annual Insurance Charge is:</p>
                    <p style="font-size:48px; font-weight:bolder; color:#00b300;">{formatted_charges}</p>
                </div>
                """, 
                unsafe_allow_html=True
            )
            
            # Insight based on Smoker status (since it's the strongest predictor)
            if smoker_yes == 1:
                st.info("⚠️ **Smoker Status Impact:** Smoking significantly increases the predicted charges. This feature has the largest coefficient in your linear model.")
            else:
                 st.info("✅ **Health Note:** Not being a smoker helps keep the estimated charges lower.")


        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# Provide instructions if the model is missing
else:
    st.warning("Please ensure the file 'best_model_for_insurance.pkl' is correctly uploaded for the application to function.")

st.write("---")
st.caption("Model based on Linear Regression using Age, BMI, and Smoker status.")
