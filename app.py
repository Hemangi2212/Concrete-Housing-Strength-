import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Concrete Strength Predictor", layout="wide")

# ----------------------------
# CUSTOM CSS FOR MODERN UI
# ----------------------------
st.markdown("""
<style>

body {
    background-color: #0d1117;
    color: white;
}

.main-container {
    background-color: #161b22;
    padding: 30px;
    border-radius: 15px;
    box-shadow: 0px 0px 20px rgba(255, 255, 255, 0.1);
}

input {
    border-radius: 10px !important;
}

</style>
""", unsafe_allow_html=True)


# ----------------------------
# LOAD MODEL
# ----------------------------
@st.cache_resource
def load_model():
    return joblib.load("AdaBoost_Best_Model.pkl")

model = load_model()


# ----------------------------
# HEADER SECTION
# ----------------------------
st.markdown("<h1 style='text-align:center; color:#ffb86c;'>🧱 Concrete Compressive Strength Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#c9d1d9;'>Enter material proportions and get instant strength prediction.</p>", unsafe_allow_html=True)

st.write("")
st.write("")

# ----------------------------
# INPUT FORM (CENTERED)
# ----------------------------
with st.container():
    st.markdown("<div class='main-container'>", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        cement = st.number_input("Cement (kg/m³)", 0.0, 600.0, 200.0)
        slag = st.number_input("Blast Furnace Slag (kg/m³)", 0.0, 360.0, 50.0)
        flyash = st.number_input("Fly Ash (kg/m³)", 0.0, 200.0, 30.0)
        water = st.number_input("Water (kg/m³)", 0.0, 250.0, 150.0)

    with col2:
        superplasticizer = st.number_input("Superplasticizer (kg/m³)", 0.0, 40.0, 5.0)
        coarse_agg = st.number_input("Coarse Aggregate (kg/m³)", 500.0, 1200.0, 900.0)
        fine_agg = st.number_input("Fine Aggregate (kg/m³)", 300.0, 1000.0, 700.0)
        age = st.number_input("Age (days)", 1, 365, 28)

    st.write("")
    st.write("")

    input_data = np.array([[cement, slag, flyash, water, superplasticizer, coarse_agg, fine_agg, age]])

    center = st.columns([4, 2, 4])[1]  
    with center:
        if st.button("🔍 Predict Strength", use_container_width=True):
            pred = model.predict(input_data)[0]
            st.success(f"💪 Predicted Concrete Strength: {pred:.2f} MPa")

    st.markdown("</div>", unsafe_allow_html=True)


st.write("")
st.write("")

# ----------------------------
# FOOTER WITH YOUR DETAILS
# ----------------------------
st.markdown("""
<hr>

<h3 style='text-align:center; color:#58a6ff;'>👩‍💻 Developer Info</h3>

<p style='text-align:center;'>
<b>Name:</b> Hemangi Ransing <br>
<b>Mobile:</b> +91-8767509860 <br>
<b>Email:</b> ransinghemangi@gmail.com <br>
<b>LinkedIn:</b> <a href='https://www.linkedin.com/in/hemangi-ransing' target='_blank'>Click Here</a>
</p>

<p style='text-align:center; color:#8b949e;'>Made with ❤️ using Streamlit</p>
""", unsafe_allow_html=True)
