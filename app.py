# app.py — VR Stroke Screening (simple, friendly, folder-aware)

import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="VR Stroke Screening", layout="centered")
st.title("🧠 VR Stroke Screening")

# -----------------------------------
# Paths (checks models/ then root; data/ then root)
# -----------------------------------
MODEL_PATHS = [
    os.path.join("models", "stroke_classifier_from_raw.pkl"),
    "stroke_classifier_from_raw.pkl",
]
DATA_CANDIDATES = [
    os.path.join("data", "vr_combined_raw.csv"),
    os.path.join("data", "all_vr_data_raw.csv"),
    os.path.join("data", "vr_feature_summary.csv"),
    "vr_combined_raw.csv",
    "all_vr_data_raw.csv",
    "vr_feature_summary.csv",
]

# -----------------------------------
# Helpers
# -----------------------------------
def exists(path: str) -> bool:
    return os.path.exists(path) and (os.path.getsize(path) > 0)

def friendly_title(name: str) -> str:
    """Prettier chart titles for common column names."""
    mapping = {
        "x": "Head Motion (X)",
        "y": "Head Motion (Y)",
        "z": "Head Motion (Z)",
        "head_x": "Head Motion (X)",
        "head_y": "Head Motion (Y)",
        "head_z": "Head Motion (Z)",
        "eye_x": "Eye Movement (X)",
        "eye_y": "Eye Movement (Y)",
        "eye_z": "Eye Movement (Z)",
        "timestamp": "Time (s)",
    }
    key = str(name).lower()
    return mapping.get(key, name.replace("_", " ").title())

def build_input_row(expected_cols, x, y, z):
    """Map sliders to model input order. Unknown features -> 0."""
    cols = expected_cols if expected_cols else ["x", "y", "z"]
    row = []
    for name in cols:
        nl = str(name).lower()
        if nl.startswith("x"):
            row.append(x)
        elif nl.startswith("y"):
            row.append(y)
        elif nl.startswith("z"):
            row.append(z)
        else:
            row.append(0.0)
    return np.array([row], dtype=float)

def choose_stroke_class_index(model, proba_array):
    """Return index in predict_proba corresponding to Stroke class."""
    classes = list(getattr(model, "classes_", [0, 1]))
    if set(classes) == {0, 1}:
        return 1
    # Look for a class name containing 'stroke' (not 'non')
    for i, c in enumerate(classes):
        s = str(c).lower()
        if "stroke" in s and "non" not in s:
            return i
    # fallback
    return 1 if len(proba_array) > 1 else 0

# -----------------------------------
# Load model & data
# -----------------------------------
@st.cache_resource
def load_model():
    for p in MODEL_PATHS:
        if exists(p):
            try:
                m = joblib.load(p)
                return m, p
            except Exception:
                pass
    return None, None

@st.cache_data
def load_data():
    for p in DATA_CANDIDATES:
        if exists(p):
            try:
                df = pd.read_csv(p)
                return df, p
            except Exception:
                pass
    return None, None

model, model_path = load_model()
if model is None:
    st.error("Model file not found. Place it at **models/stroke_classifier_from_raw.pkl** (preferred) or repo root.")
    st.stop()
st.caption(f"Loaded model: `{model_path}`")

df, df_path = load_data()
if df is not None:
    st.caption(f"Loaded data: `{df_path}`")

# -----------------------------------
# Quick Prediction (sliders)
# -----------------------------------
st.header("Quick Check")

# Decide expected feature names
if hasattr(model, "feature_names_in_"):
    expected = list(model.feature_names_in_)
else:
    expected = ["x", "y", "z"]

# Infer slider ranges from data if possible
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

if st.button("Predict"):
    X = build_input_row(expected, x_val, y_val, z_val)
    yhat = model.predict(X)[0]

    # Friendly label
    if hasattr(model, "classes_") and set(model.classes_) == {0, 1}:
        label = "Stroke" if int(yhat) == 1 else "Non-Stroke"
    else:
        label = str(yhat)

    st.subheader(f"Result: {label}")

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        idx = choose_stroke_class_index(model, proba)
        pct = float(proba[idx])
        st.progress(pct)
        st.write(f"Stroke probability: {pct * 100:.1f}%")

st.divider()

# -----------------------------------
# Charts (auto-detect numeric columns, skip binary labels)
# -----------------------------------
st.header("Charts")
if df is None:
    st.info("Add a CSV in **data/** (e.g., vr_combined_raw.csv) to see charts.")
else:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Remove binary label-like columns (0/1)
    pruned = []
    for col in numeric_cols:
        vals = df[col].dropna().unique()
        if len(vals) <= 3 and set(vals).issubset({0, 1}):
            # binary label → skip from plotting
            continue
        pruned.append(col)

    if not pruned:
        st.info("Found only binary columns (like labels). Add motion columns to plot.")
    else:
        # Choose time column if present
        time_col = None
        for cand in ["timestamp", "time", "t", "frame"]:
            if cand in df.columns:
                time_col = cand
                break

        # Preferred plotting order, then fill with any other numeric columns
        preferred = [c for c in ["x", "y", "z", "head_x", "head_y", "head_z",
                                 "eye_x", "eye_y", "eye_z"] if c in pruned]
        others = [c for c in pruned if c not in preferred]
        to_plot = (preferred + others)[:6]  # cap to avoid huge render cost

        for col in to_plot:
            fig, ax = plt.subplots()
            if time_col:
                ax.plot(df[time_col], df[col])
                ax.set_xlabel(friendly_title(time_col))
            else:
                ax.plot(df.index, df[col])
                ax.set_xlabel("Index")
            ax.set_ylabel("Value")
            ax.set_title(friendly_title(col))
            st.pyplot(fig)

# -----------------------------------
# Footer
# -----------------------------------
st.caption("Demo — for education and research only.")
