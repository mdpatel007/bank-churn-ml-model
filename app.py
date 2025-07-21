import streamlit as st
import pandas as pd
import numpy as np
import pickle  # For loading model and scaler

# --- Load Model and Scaler ---
# Using Streamlit's caching feature to prevent reloading every time
@st.cache_resource
def load_resources():
    try:
        with open('churn_model.pkl', 'rb') as file:
            model = pickle.load(file)
        with open('scaler.pkl', 'rb') as file:
            scaler = pickle.load(file)
        with open('column_names.pkl', 'rb') as file:
            column_names = pickle.load(file)
        return model, scaler, column_names
    except FileNotFoundError:
        st.error("Model, scaler, or column names file not found. Make sure 'churn_model.pkl', 'scaler.pkl', and 'column_names.pkl' are in the same directory.")
        st.stop()  # Stop the app if files are missing

model, scaler, column_names = load_resources()

# --- App Heading ---
st.set_page_config(page_title="Bank Customer Churn Prediction", page_icon="🏦")
st.title("🏦 Bank Customer Churn Prediction")
st.markdown("Enter customer details to predict if they will churn.")
st.markdown("---")

# --- User Input Fields ---
st.header("Customer Details Input")

col1, col2 = st.columns(2)

with col1:
    credit_score = st.slider("Credit Score", 350, 850, 600)
    age = st.slider("Age", 18, 92, 35)
    tenure = st.slider("Tenure (Years)", 0, 10, 5)
    balance = st.number_input("Bank Balance", 0.0, 250000.0, 50000.0, step=1000.0)
    num_products = st.slider("Number of Products", 1, 4, 1)

with col2:
    has_credit_card = st.radio("Has Credit Card?", ("Yes", "No"))
    is_active_member = st.radio("Is Active Member?", ("Yes", "No"))
    estimated_salary = st.number_input("Estimated Salary", 0.0, 200000.0, 80000.0, step=1000.0)
    country = st.selectbox("Country", ("France", "Germany", "Spain"))
    gender = st.selectbox("Gender", ("Male", "Female"))

# Convert inputs to model-understandable format
data = {
    'credit_score': credit_score,
    'age': age,
    'tenure': tenure,
    'balance': balance,
    'products_number': num_products,
    'has_credit_card': 1 if has_credit_card == "Yes" else 0,
    'is_active_member': 1 if is_active_member == "Yes" else 0,
    'estimated_salary': estimated_salary,
    'country_Germany': 1 if country == "Germany" else 0,
    'country_Spain': 1 if country == "Spain" else 0,
    'gender_Male': 1 if gender == "Male" else 0
}

# Ensure the order of features matches the training data
# Create a DataFrame with a single row based on user inputs
input_df = pd.DataFrame([data])

# Add missing columns (if any) with 0 and reorder them to match training order
final_input_df = pd.DataFrame(0, index=[0], columns=column_names)
for col in input_df.columns:
    if col in final_input_df.columns:
        final_input_df[col] = input_df[col]

# --- Preprocessing the Input ---
# Numerical columns to scale (ensure these match your original notebook)
numerical_cols_to_scale = ['credit_score', 'balance', 'estimated_salary']

# Apply scaling to the numerical columns of the input DataFrame
final_input_df[numerical_cols_to_scale] = scaler.transform(final_input_df[numerical_cols_to_scale])

# --- Prediction Button ---
st.markdown("---")
if st.button("Predict Churn"):
    prediction = model.predict(final_input_df)
    prediction_proba = model.predict_proba(final_input_df)[:, 1]  # Probability of churning

    if prediction[0] == 1:
        st.warning(f"**Prediction: This customer is likely to CHURN!** (Probability: {prediction_proba[0]:.2f}) 😟")
        st.info("Consider proactive retention strategies for this customer.")
    else:
        st.success(f"**Prediction: This customer is likely NOT to churn.** (Probability: {prediction_proba[0]:.2f}) 😊")
        st.info("Great! Keep them engaged.")

    st.markdown(f"*(Note: Probability > 0.5 typically indicates churn for a binary classifier)*")
st.markdown("---")
st.caption("Developed by You | Bank Churn Prediction Model")
