import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

# -------------------------------------------------------------------
# BASIC SETUP
# -------------------------------------------------------------------
st.set_page_config(page_title="VR Stroke Screening App", layout="centered")
st.title("🧠 VR Stroke Screening Tool")
st.markdown("This tool analyzes head and eye motion to estimate whether a pattern looks more like a stroke or a non-stroke case.")

# -------------------------------------------------------------------
# LOAD MODEL
# -------------------------------------------------------------------
@st.cache_resource
def load_model():
    model_path = "stroke_classifier_from_raw.pkl"
    if not os.path.exists(model_path):
        st.error("Model file not found. Please upload 'stroke_classifier_from_raw.pkl'.")
        return None
    model = joblib.load(model_path)
    return model

model = load_model()

# -------------------------------------------------------------------
# LOAD DATA (optional visualization dataset)
# -------------------------------------------------------------------
@st.cache_data
def load_data():
    for name in ["vr_combined_raw.csv", "all_vr_data_raw.csv", "vr_feature_summary.csv"]:
        if os.path.exists(name):
            df = pd.read_csv(name)
            return df
    return None

df = load_data()

# -------------------------------------------------------------------
# PREDICTION SECTION
# -------------------------------------------------------------------
st.header("🩺 Try a Quick Test")

col1, col2, col3 = st.columns(3)
with col1:
    x = st.slider("Head Motion (X)", -20.0, 20.0, 0.0)
with col2:
    y = st.slider("Head Motion (Y)", -50.0, 150.0, 0.0)
with col3:
    z = st.slider("Head Motion (Z)", -20.0, 20.0, 0.0)

if st.button("Run Test"):
    if model is None:
        st.error("Model not loaded.")
    else:
        user_input = pd.DataFrame([[x, y, z]], columns=["x", "y", "z"])
        prediction = model.predict(user_input)[0]
        proba = model.predict_proba(user_input)[0][1]  # stroke prob

        if prediction == 1 or proba > 0.5:
            st.error(f"⚠️ Result: Stroke Likely ({proba*100:.1f}% probability)")
        else:
            st.success(f"✅ Result: Non-Stroke ({(1-proba)*100:.1f}% probability)")

# -------------------------------------------------------------------
# VISUALIZATION SECTION
# -------------------------------------------------------------------
st.header("📊 Motion Data Overview")
if df is not None:
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    if len(numeric_cols) >= 2:
        x_axis = st.selectbox("X-axis", numeric_cols, index=0)
        y_axis = st.selectbox("Y-axis", numeric_cols, index=min(1, len(numeric_cols)-1))
        fig, ax = plt.subplots()
        ax.scatter(df[x_axis], df[y_axis], alpha=0.4, c="teal")
        ax.set_xlabel(x_axis)
        ax.set_ylabel(y_axis)
        ax.set_title("Motion Pattern Distribution")
        st.pyplot(fig)
    else:
        st.info("Not enough numeric columns for plotting.")
else:
    st.info("No motion data file found. Add one to visualize patterns.")

# -------------------------------------------------------------------
# FOOTER
# -------------------------------------------------------------------
st.caption("Demo version – for educational and research use only.")
