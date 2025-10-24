import pandas as pd
import numpy as np
import streamlit as st
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load data for context
data = pd.read_csv('Expresso_churn_dataset.csv')

# ----------------- HEADER -----------------
st.markdown("<h1 style='color:#114232; text-align:center; font-size:60px; font-family:Monospace'>EXPRESS CHURN APP</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='margin:-30px; color:#87A922; text-align:center; font-family:cursive'>Built by Modupe Oshinjirin</h4>", unsafe_allow_html=True)
st.image('pngwing.com (13).png')

# ----------------- BACKGROUND -----------------
st.markdown("<h2 style='color:#FF9800; text-align:center; font-family:montserrat'>Background Of Study</h2>", unsafe_allow_html=True)
st.markdown("""
Expresso is a leading African telecommunications company operating in Mauritania and Senegal.  
The company provides a wide range of mobile and data services to millions of customers.

Like many telecom companies, Expresso faces a key business challenge — **customer churn**, where users stop using their services.  
To reduce churn, this project uses **machine learning** to predict whether a customer is likely to leave the network based on their usage patterns and behavior.

Use this simple app to input customer details and instantly see the predicted churn status!
""")


st.sidebar.image('pngwing.com (3).png')

# ----------------- DATA DISPLAY -----------------
st.divider()
st.header('Project Data')
st.dataframe(data.head(50), use_container_width=True)
st.info("Only the first 50 rows are displayed for preview. Full dataset is too large to render.")


# Load model
model = joblib.load('expresso_churn_model.pkl')

# Define feature names in the same order as training
feature_names = [
    'REGION', 'MONTANT', 'FREQUENCE_RECH', 'REVENUE', 'ARPU_SEGMENT',
    'FREQUENCE', 'DATA_VOLUME', 'ON_NET', 'ORANGE', 'TIGO',
    'ZONE1', 'ZONE2', 'MRG', 'FREQ_TOP_PACK', 'TENURE'
]


# ----------------- SIDEBAR INPUT -----------------
st.sidebar.header("🔹 Customer Information")


region = st.sidebar.number_input("REGION", min_value=0)
montant = st.sidebar.number_input("MONTANT", min_value=0.0)
frequence_rech = st.sidebar.number_input("FREQUENCE_RECH", min_value=0)
revenue = st.sidebar.number_input("REVENUE", min_value=0.0)
arpu_segment = st.sidebar.number_input("ARPU_SEGMENT", min_value=0.0)
frequence = st.sidebar.number_input("FREQUENCE", min_value=0)
data_volume = st.sidebar.number_input("DATA_VOLUME", min_value=0.0)
on_net = st.sidebar.number_input("ON_NET", min_value=0)
orange = st.sidebar.number_input("ORANGE", min_value=0)
tigo = st.sidebar.number_input("TIGO", min_value=0)
zone1 = st.sidebar.selectbox("ZONE1", [0, 1])
zone2 = st.sidebar.selectbox("ZONE2", [0, 1])
mrg = st.sidebar.number_input("MRG", min_value=0)
freq_top_pack = st.sidebar.number_input("FREQ_TOP_PACK", min_value=0)
tenure = st.sidebar.number_input("TENURE", min_value=0)

# ----------------- INPUT DATA -----------------
input_data = pd.DataFrame([[
    region, montant, frequence_rech, revenue, arpu_segment,
    frequence, data_volume, on_net, orange, tigo,
    zone1, zone2, mrg, freq_top_pack, tenure
]], columns=feature_names)

# ----------------- PREDICTION -----------------
st.write("---")
st.subheader("🔍 Churn Prediction Result")

if st.button("Predict Churn"):
    try:
        prediction = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error(f"⚠️ The customer is likely to **CHURN** (probability: {prob:.2f})")
        else:
            st.success(f"✅ The customer is likely to **STAY** (probability: {prob:.2f})")
    except Exception as e:
        st.error(f"An error occurred: {e}")

st.write("---")
st.markdown("<p style='text-align:center; color:gray;'>Expresso Churn ML App © 2025</p>", unsafe_allow_html=True)