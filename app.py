# app.py — VR Stroke Screening (respects models/ and data/ folders)
import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="VR Stroke Screening", layout="centered")
st.title("🧠 VR Stroke Screening")

# -----------------------------
# Paths to check (in order)
# -----------------------------
MODEL_PATHS = [
    os.path.join("models", "stroke_classifier_from_raw.pkl"),
    "stroke_classifier_from_raw.pkl",
]

DATA_CANDIDATES = [
    os.path.join("data", "vr_combined_raw.csv"),
    "vr_combined_raw.csv",
    os.path.join("data", "all_vr_data_raw.csv"),
    "all_vr_data_raw.csv",
    os.path.join("data", "vr_feature_summary.csv"),
    "vr_feature_summary.csv",
]

# -----------------------------
# Load model
# -----------------------------
@st.cache_resource
def load_model():
    for p in MODEL_PATHS:
        if os.path.exists(p):
            try:
                m = joblib.load(p)
                return m, p
            except Exception:
                pass
    return None, None

model, model_path = load_model()
if model is None:
    st.error("Model file not found. Place it at **models/stroke_classifier_from_raw.pkl** (preferred) or repo root.")
    st.stop()
st.caption(f"Loaded model: `{model_path}`")

# -----------------------------
# Load data (optional, for charts)
# -----------------------------
@st.cache_data
def load_data():
    for p in DATA_CANDIDATES:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                return df, p
            except Exception:
                pass
    return None, None

df, df_path = load_data()
if df is not None:
    st.caption(f"Loaded data: `{df_path}`")

# -----------------------------
# Quick prediction (sliders)
# If model expects x,y,z → use x,y,z sliders.
# Otherwise still use x,y,z and map best-effort.
# -----------------------------
st.header("Quick Check")

# Decide expected feature names
if hasattr(model, "feature_names_in_"):
    expected = list(model.feature_names_in_)
else:
    expected = ["x", "y", "z"]

# Slider ranges: infer from data if possible, else defaults
def infer_range(col, default_lo, default_hi):
    if df is not None and col in df.columns and df[col].notna().any():
        q1, q99 = np.nanpercentile(df[col], [1, 99])
        if np.isfinite(q1) and np.isfinite(q99) and q1 < q99:
            return float(q1), float(q99)
    return default_lo, default_hi

x_lo, x_hi = infer_range("x", -20.0, 20.0)
y_lo, y_hi = infer_range("y", -50.0, 150.0)
z_lo, z_hi = infer_range("z", -20.0, 20.0)

c1, c2, c3 = st.columns(3)
with c1:
    x_val = st.slider("X-axis motion", float(x_lo), float(x_hi), float((x_lo + x_hi) / 2))
with c2:
    y_val = st.slider("Y-axis motion", float(y_lo), float(y_hi), float((y_lo + y_hi) / 2))
with c3:
    z_val = st.slider("Z-axis motion", float(z_lo), float(z_hi), float((z_lo + z_hi) / 2))

def build_input_row(expected_cols, x, y, z):
    row = []
    for name in (expected_cols if expected_cols else ["x", "y", "z"]):
        nl = str(name).lower()
        if nl.startswith("x"):
            row.append(x)
        elif nl.startswith("y"):
            row.append(y)
        elif nl.startswith("z"):
            row.append(z)
        else:
            # unseen engineered feature → 0
            row.append(0.0)
    return np.array([row], dtype=float)

if st.button("Predict"):
    X = build_input_row(expected, x_val, y_val, z_val)
    yhat = model.predict(X)[0]

    # map label to friendly text
    if hasattr(model, "classes_") and set(model.classes_) == {0, 1}:
        label = "Stroke" if int(yhat) == 1 else "Non-Stroke"
    else:
        label = str(yhat)

    st.subheader(f"Result: {label}")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        classes = list(getattr(model, "classes_", [0, 1]))
        # index for "stroke" class
        stroke_idx = 1 if set(classes) == {0, 1} else (
            classes.index(next((c for c in classes if "stroke" in str(c).lower() and "non" not in str(c).lower()), classes[-1]))
            if classes else 1
        )
        pct = float(proba[stroke_idx])
        st.progress(pct)
        st.write(f"Stroke probability: {pct * 100:.1f}%")

st.divider()

# -----------------------------
# Charts (auto-detect numeric columns)
# -----------------------------
st.header("Charts")
if df is None:
    st.info("Add a CSV in **data/** (e.g., vr_combined_raw.csv) to see charts.")
else:
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not num_cols:
        st.info("No numeric columns found to plot.")
    else:
        # Choose time column if present
        time_col = None
        for cand in ["timestamp", "time", "t", "frame"]:
            if cand in df.columns:
                time_col = cand
                break

        # Plot up to 6 useful series (x,y,z + any others)
        preferred_order = [c for c in ["x", "y", "z", "head_x", "head_y", "head_z", "eye_x", "eye_y", "eye_z"] if c in num_cols]
        others = [c for c in num_cols if c not in preferred_order]
        to_plot = (preferred_order + others)[:6]

        for col in to_plot:
            fig, ax = plt.subplots()
            if time_col:
                ax.plot(df[time_col], df[col])
                ax.set_xlabel(time_col)
            else:
                ax.plot(df.index, df[col])
                ax.set_xlabel("Index")
            ax.set_title(col)
            ax.set_ylabel("Value")
            st.pyplot(fig)
