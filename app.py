# app.py — VR Stroke Screening (auto-matches model features; simple, friendly UI)
# Run: streamlit run app.py

import os, io, glob
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt

from typing import Optional, Tuple
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="VR Stroke Screening", layout="wide")
st.title("VR Stroke Screening")

# -----------------------------
# Paths
# -----------------------------
DATA_DIRS = ["data", "."]
MODEL_PATHS = [
    os.path.join("models", "stroke_classifier_from_raw.pkl"),
    "stroke_classifier_from_raw.pkl",
]

CSV_CANDIDATES = ["vr_combined_raw.csv", "vr_feature_summary.csv", "all_vr_data_raw.csv"]
TXT_NAMES = [
    "HeadPositionS.txt","HeadRotationS.txt","LeftEyeRotationS.txt","RightEyeRotationS.txt",
    "HeadPositionNS.txt","HeadRotationNS.txt","LeftEyeRotationNS.txt","RightEyeRotationNS.txt",
]

# -----------------------------
# Utils
# -----------------------------
def exists(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0

def try_load_model():
    for p in MODEL_PATHS:
        if exists(p):
            try:
                return joblib.load(p), p
            except Exception:
                pass
    return None, None

def try_load_csv():
    for base in DATA_DIRS:
        for name in CSV_CANDIDATES:
            p = os.path.join(base, name)
            if exists(p):
                try:
                    df = pd.read_csv(p)
                    return df, p
                except Exception:
                    pass
    return None, None

def parse_record_line(line: str) -> Optional[Tuple[float, float, float, float]]:
    try:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            return None
        sec = np.nan
        if ":" in parts[0]:
            hh, mm, ss = parts[0].split(":")
            sec = int(hh)*3600 + int(mm)*60 + float(ss)
        x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
        return sec, x, y, z
    except Exception:
        return None

def read_vr_txt(fp: str) -> Optional[pd.DataFrame]:
    rows = []
    with open(fp, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            rec = parse_record_line(line)
            if rec: rows.append(rec)
    if not rows: return None
    df = pd.DataFrame(rows, columns=["timestamp_raw","x","y","z"])
    if df["timestamp_raw"].isna().all():
        df["timestamp"] = np.arange(len(df))/60.0
    else:
        start = df["timestamp_raw"].dropna().iloc[0]
        df["timestamp"] = df["timestamp_raw"] - start
    return df.drop(columns=["timestamp_raw"])

def load_txt_bundle():
    bundle = {}
    for base in DATA_DIRS:
        if not os.path.isdir(base): continue
        # expected first
        for name in TXT_NAMES:
            p = os.path.join(base, name)
            if exists(p):
                d = read_vr_txt(p)
                if d is not None and not d.empty:
                    bundle[os.path.splitext(name)[0]] = d
        # any other txt
        for p in glob.glob(os.path.join(base, "*.txt")):
            base_name = os.path.basename(p)
            if base_name in TXT_NAMES: continue
            if exists(p):
                d = read_vr_txt(p)
                if d is not None and not d.empty:
                    bundle[os.path.splitext(base_name)[0]] = d
    return bundle

def assemble_table(bundle: dict) -> Optional[pd.DataFrame]:
    if not bundle: return None
    first_key = sorted(bundle.keys())[0]
    base = bundle[first_key].copy()
    base["timestamp"] = base["timestamp"].astype(float)
    wide = pd.DataFrame({"timestamp": base["timestamp"]})
    def role(k: str) -> str: return "eye" if "eye" in k.lower() else "head"
    for key, df in bundle.items():
        r = role(key)
        n = min(len(wide), len(df))
        if n == 0: continue
        wide.loc[:n-1, f"{r}_x"] = df["x"].values[:n]
        wide.loc[:n-1, f"{r}_y"] = df["y"].values[:n]
        wide.loc[:n-1, f"{r}_z"] = df["z"].values[:n]
    for c in ["head_x","head_y","head_z","eye_x","eye_y","eye_z"]:
        if c not in wide.columns: wide[c] = np.nan
    wide["label"] = "Unknown"
    return wide[["timestamp","head_x","head_y","head_z","eye_x","eye_y","eye_z","label"]]

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    out = {}
    cols = ["head_x","head_y","head_z","eye_x","eye_y","eye_z"]
    for c in cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            out[f"{c}_mean"] = s.mean()
            out[f"{c}_std"]  = s.std()
            out[f"{c}_mad"]  = (s - s.mean()).abs().mean()
            out[f"{c}_max"]  = s.max()
            out[f"{c}_min"]  = s.min()
    if "timestamp" in df.columns:
        t = pd.to_numeric(df["timestamp"], errors="coerce").values
        for c in cols:
            if c in df.columns:
                vals = pd.to_numeric(df[c], errors="coerce").values
                dv = np.diff(vals)
                dt = np.diff(t[:len(vals)])
                with np.errstate(divide="ignore", invalid="ignore"):
                    vel = np.where(dt != 0, dv/dt, np.nan)
                if vel.size > 1:
                    out[f"{c}_vel_mean"] = np.nanmean(vel)
                    out[f"{c}_vel_std"]  = np.nanstd(vel)
    return pd.DataFrame([out])

def align_to_model(model, feats_df: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
        return feats_df.reindex(columns=cols, fill_value=0.0).to_numpy()
    return feats_df.select_dtypes(include=[np.number]).to_numpy()

def map_label(model, yhat):
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
        if set(classes) == {0,1}:
            return "Stroke" if int(yhat) == 1 else "Non-Stroke"
        return str(yhat)
    return "Stroke" if int(yhat) == 1 else "Non-Stroke"

# -----------------------------
# Load model + data
# -----------------------------
model, model_path = try_load_model()
if model is None:
    st.error("Model file not found. Add models/stroke_classifier_from_raw.pkl")
    st.stop()

csv_df, csv_path = try_load_csv()
if csv_df is not None:
    df_raw = csv_df.copy()
else:
    bundle = load_txt_bundle()
    df_raw = assemble_table(bundle) if bundle else pd.DataFrame(columns=["timestamp","head_x","head_y","head_z","eye_x","eye_y","eye_z","label"])

for c in ["head_x","head_y","head_z","eye_x","eye_y","eye_z"]:
    if c not in df_raw.columns: df_raw[c] = np.nan
if "timestamp" not in df_raw.columns:
    df_raw["timestamp"] = np.arange(len(df_raw))/60.0
if "label" not in df_raw.columns:
    df_raw["label"] = "Unknown"

# -----------------------------
# Decide input mode from model's expected features
# -----------------------------
if hasattr(model, "feature_names_in_"):
    expected = list(model.feature_names_in_)
else:
    # try to infer from training data you used previously
    expected = ["x", "y", "z"]

simple_numeric_mode = False
if len(expected) <= 6 and all(k.lower() in {"x","y","z","x1","y1","z1"} or k.lower() in {"x_axis","y_axis","z_axis"} for k in expected):
    simple_numeric_mode = True
elif expected == ["x","y","z"]:
    simple_numeric_mode = True

# -----------------------------
# TOP: Quick Prediction (user-friendly)
# -----------------------------
st.header("Quick Check")

if simple_numeric_mode:
    # Infer slider ranges from data if possible
    ref = None
    for col in ["x","y","z"]:
        if col in df_raw.columns:
            ref = df_raw
            break
    if ref is None:
        # try typical ranges from your old app
        x_min, x_max = -20.0, 20.0
        y_min, y_max = -50.0, 150.0
        z_min, z_max = -20.0, 20.0
    else:
        # robust percentiles if present
        def rng(c, lo, hi):
            if c in ref.columns and ref[c].notna().any():
                q1, q99 = np.nanpercentile(ref[c], [1, 99])
                if np.isfinite(q1) and np.isfinite(q99) and q1 != q99:
                    return float(q1), float(q99)
            return lo, hi
        x_min, x_max = rng("x", -20.0, 20.0)
        y_min, y_max = rng("y", -50.0, 150.0)
        z_min, z_max = rng("z", -20.0, 20.0)

    c1, c2, c3 = st.columns(3)
    with c1:
        x_val = st.slider("X-axis motion", float(x_min), float(x_max), float((x_min+x_max)/2.0))
    with c2:
        y_val = st.slider("Y-axis motion", float(y_min), float(y_max), float((y_min+y_max)/2.0))
    with c3:
        z_val = st.slider("Z-axis motion", float(z_min), float(z_max), float((z_min+z_max)/2.0))

    if st.button("Predict"):
        # Build input exactly as model expects
        cols = expected if expected else ["x","y","z"]
        row = []
        for name in cols:
            nl = name.lower()
            if "x" == nl or "x_axis" == nl or nl.startswith("x"):
                row.append(x_val)
            elif "y" == nl or "y_axis" == nl or nl.startswith("y"):
                row.append(y_val)
            elif "z" == nl or "z_axis" == nl or nl.startswith("z"):
                row.append(z_val)
            else:
                row.append(0.0)
        X = np.array([row], dtype=float)

        yhat = model.predict(X)[0]
        label = map_label(model, yhat)
        st.subheader(f"Result: {label}")
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            classes = list(getattr(model, "classes_", [0,1]))
            stroke_idx = 1 if set(classes) == {0,1} else classes.index(next((c for c in classes if "stroke" in str(c).lower() and "non" not in str(c).lower()), classes[-1]))
            pct = float(proba[stroke_idx])
            st.progress(pct)
            st.write(f"Stroke probability: {pct*100:.1f}%")
else:
    # Fallback: engineered feature mode (kept simple)
    st.write("Move the controls and check the result.")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        head_stability = st.slider("Head stability", 0.0, 1.0, 0.7, 0.01)
    with col2:
        eye_fixation   = st.slider("Eye fixation", 0.0, 1.0, 0.7, 0.01)
    with col3:
        head_range     = st.slider("Head movement range", 0.0, 2.0, 0.5, 0.01)
    with col4:
        eye_range      = st.slider("Eye movement range", 0.0, 1.0, 0.2, 0.01)

    sensitivity = st.slider("Sensitivity", 0.5, 5.0, 2.5, 0.1)

    def clamp(x, lo, hi): return max(lo, min(hi, x))

    if st.button("Predict"):
        base = compute_features(df_raw)
        # ensure all expected columns exist
        if hasattr(model, "feature_names_in_"):
            for c in model.feature_names_in_:
                if c not in base.columns:
                    base[c] = 0.0
        demo = base.copy()
        for col in demo.columns:
            lc = col.lower()
            if ("std" in lc or "vel_std" in lc) and ("head" in lc):
                demo[col] = demo[col].astype(float) * (1.0 + sensitivity*(1.0 - head_stability))
            if ("std" in lc or "vel_std" in lc) and ("eye" in lc):
                demo[col] = demo[col].astype(float) * (1.0 + sensitivity*(1.0 - eye_fixation))
            if "mean" in lc and "head" in lc:
                demo[col] = demo[col].astype(float) + 0.1*sensitivity*(0.5 - head_stability)
            if "mean" in lc and "eye" in lc:
                demo[col] = demo[col].astype(float) + 0.1*sensitivity*(0.5 - eye_fixation)
            if "max" in lc and "head" in lc:
                demo[col] = demo[col].astype(float) * clamp(head_range, 0.1, 2.0)
            if "min" in lc and "head" in lc:
                demo[col] = demo[col].astype(float) * clamp(head_range, 0.1, 2.0)
            if "max" in lc and "eye" in lc:
                demo[col] = demo[col].astype(float) * clamp(eye_range, 0.05, 1.0)
            if "min" in lc and "eye" in lc:
                demo[col] = demo[col].astype(float) * clamp(eye_range, 0.05, 1.0)
            if "vel_mean" in lc and "head" in lc:
                demo[col] = demo[col].astype(float) * (0.5 + 0.25*sensitivity*head_range)
            if "vel_mean" in lc and "eye" in lc:
                demo[col] = demo[col].astype(float) * (0.5 + 0.25*sensitivity*eye_range)

        if hasattr(model, "feature_names_in_"):
            X = demo.reindex(columns=list(model.feature_names_in_), fill_value=0.0).to_numpy()
        else:
            X = demo.select_dtypes(include=[np.number]).to_numpy()

        yhat = model.predict(X)[0]
        label = map_label(model, yhat)
        st.subheader(f"Result: {label}")
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            classes = list(getattr(model, "classes_", [0,1]))
            stroke_idx = 1 if set(classes) == {0,1} else classes.index(next((c for c in classes if "stroke" in str(c).lower() and "non" not in str(c).lower()), classes[-1]))
            pct = float(proba[stroke_idx])
            st.progress(pct)
            st.write(f"Stroke probability: {pct*100:.1f}%")

st.divider()

# -----------------------------
# Tabs (kept simple)
# -----------------------------
tab1, tab2 = st.tabs(["Data Preview", "Charts"])

with tab1:
    st.subheader("Data Preview")
    st.dataframe(df_raw.head(50), use_container_width=True)

with tab2:
    st.subheader("Charts")
    if "timestamp" in df_raw.columns:
        t = pd.to_numeric(df_raw["timestamp"], errors="coerce")
        for col in ["head_x","head_y","head_z","eye_x","eye_y","eye_z"]:
            if col in df_raw.columns and df_raw[col].notna().any():
                fig, ax = plt.subplots()
                ax.plot(t, pd.to_numeric(df_raw[col], errors="coerce"))
                ax.set_title(col); ax.set_xlabel("Time (s)"); ax.set_ylabel("Value")
                st.pyplot(fig)
