import streamlit as st
import pandas as pd
import joblib

# Load trained model (expecting 7 features)
model = joblib.load("model.joblib")

st.set_page_config(page_title="Telco Churn Predictor", layout="centered")
st.title("📞 Telco Churn Prediction (Top 7 Features)")
st.markdown("Predict customer churn using the most impactful 7 features.")

# Mapping dictionaries
contract_map = {"Month-to-month": 0, "One year": 1, "Two year": 2}
internet_service_map = {"No": 0, "DSL": 1, "Fiber optic": 2}
three_option_map = {"No": 0, "Yes": 1, "No internet": 2}

# Input form
def get_user_input():
    tenure = st.slider("Tenure (in months)", 0, 72, 12)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=2500.0)
    
    contract = contract_map[st.selectbox("Contract Type", list(contract_map.keys()))]
    internet_service = internet_service_map[st.selectbox("Internet Service", list(internet_service_map.keys()))]
    
    online_security = three_option_map[st.selectbox("Online Security", list(three_option_map.keys()))]
    tech_support = three_option_map[st.selectbox("Tech Support", list(three_option_map.keys()))]

    data = [[
        tenure, monthly_charges, total_charges,
        contract, internet_service, online_security, tech_support
    ]]
    
    columns = [
        'tenure', 'MonthlyCharges', 'TotalCharges',
        'Contract', 'InternetService', 'OnlineSecurity', 'TechSupport'
    ]
    
    return pd.DataFrame(data, columns=columns)

# Get input
user_input = get_user_input()

# Predict
if st.button("Predict Churn"):
    try:
        prediction = model.predict(user_input.values)[0]
        probability = model.predict_proba(user_input.values)[0][1]

        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"🚨 This customer is **likely to churn**.\n\n**Probability: {probability:.2f}**")
        else:
            st.success(f"✅ This customer is **not likely to churn**.\n\n**Confidence: {1 - probability:.2f}**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
