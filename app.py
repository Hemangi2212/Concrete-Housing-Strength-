import streamlit as st
import numpy as np
import joblib

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Concrete Strength Prediction",
    layout="centered",
)

# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_model():
    return joblib.load("AdaBoost_Best_Model.pkl")

model = load_model()

# ----------------------------
# APP HEADER
# ----------------------------
st.markdown(
    """
    <h2 style='text-align:center; color:#4A4A4A;'>
        🧱 Concrete Compressive Strength Predictor
    </h2>
    <p style='text-align:center; color:gray;'>
        Enter material quantities to predict concrete strength
    </p>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# SIDEBAR INPUTS
# ----------------------------
st.sidebar.header("🔧 Input Features")

cement = st.sidebar.number_input("Cement (kg/m³)", 0.0, 600.0, 200.0)
slag = st.sidebar.number_input("Blast Furnace Slag (kg/m³)", 0.0, 360.0, 50.0)
flyash = st.sidebar.number_input("Fly Ash (kg/m³)", 0.0, 200.0, 30.0)
water = st.sidebar.number_input("Water (kg/m³)", 0.0, 250.0, 150.0)
superplasticizer = st.sidebar.number_input("Superplasticizer (kg/m³)", 0.0, 40.0, 5.0)
coarse_agg = st.sidebar.number_input("Coarse Aggregate (kg/m³)", 500.0, 1200.0, 900.0)
fine_agg = st.sidebar.number_input("Fine Aggregate (kg/m³)", 300.0, 1000.0, 700.0)
age = st.sidebar.number_input("Age (days)", 1, 365, 28)

# Combine inputs
input_data = np.array([[cement, slag, flyash, water, superplasticizer, coarse_agg, fine_agg, age]])

# ----------------------------
# PREDICTION
# ----------------------------
st.markdown("## 🔍 Prediction")

if st.button("Predict Strength"):
    pred = model.predict(input_data)[0]
    st.success(f"💪 Predicted Concrete Strength: **{pred:.2f} MPa**")

# ----------------------------
# FOOTER
# ----------------------------
st.markdown(
    "<hr><p style='text-align:center; color:grey;'>Made with ❤️ in Streamlit</p>",
    unsafe_allow_html=True
)
