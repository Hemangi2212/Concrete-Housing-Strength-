import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="Prediction App",
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
        🚀 Machine Learning Prediction App
    </h2>
    <p style='text-align:center; color:gray;'>
        Enter the values below to get predictions from the AdaBoost Model
    </p>
    """,
    unsafe_allow_html=True
)

# ----------------------------
# INPUT SIDEBAR
# ----------------------------
st.sidebar.header("🔧 Input Features")

# 👉 Replace these with your model’s actual feature names
# Example feature names (update as per your dataset)
feature_1 = st.sidebar.number_input("Feature 1", 0.0, 1000.0, 10.0)
feature_2 = st.sidebar.number_input("Feature 2", 0.0, 1000.0, 20.0)
feature_3 = st.sidebar.number_input("Feature 3", 0.0, 1000.0, 30.0)
feature_4 = st.sidebar.number_input("Feature 4", 0.0, 1000.0, 40.0)

# Make final input array
input_data = np.array([[feature_1, feature_2, feature_3, feature_4]])

# ----------------------------
# PREDICT BUTTON
# ----------------------------
st.markdown("## 🔍 Prediction")

if st.button("Predict"):
    pred = model.predict(input_data)[0]
    st.success(f"🎉 **Prediction: {pred}**")

    # Optional probability
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_data)[0][1]
        st.info(f"📊 Probability: {proba:.2f}")

# ----------------------------
# FOOTER
# ----------------------------
st.markdown(
    "<hr><p style='text-align:center; color:grey;'>Made with ❤️ using Streamlit</p>",
    unsafe_allow_html=True
)
