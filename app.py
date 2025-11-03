# app.py — VR Stroke Screening (simple, user-friendly)
# Run: streamlit run app.py

import os, io, glob
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt

from typing import Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="VR Stroke Screening", layout="wide")
st.title("VR Stroke Screening")

DATA_DIR = os.path.join(os.getcwd(), "data")
MODEL_PATH = os.path.join("models", "stroke_classifier_from_raw.pkl")
CSV_CANDIDATES = ["vr_feature_summary.csv", "vr_combined_raw.csv", "all_vr_data_raw.csv"]
TXT_NAMES = [
    "HeadPositionS.txt","HeadRotationS.txt","LeftEyeRotationS.txt","RightEyeRotationS.txt",
    "HeadPositionNS.txt","HeadRotationNS.txt","LeftEyeRotationNS.txt","RightEyeRotationNS.txt",
]

def exists(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0

def load_first_csv() -> Optional[pd.DataFrame]:
    for name in CSV_CANDIDATES:
        p = os.path.join(DATA_DIR, name)
        if exists(p):
            return pd.read_csv(p)
    return None

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

def load_txt_bundle() -> dict:
    bundle = {}
    if not os.path.isdir(DATA_DIR): return bundle
    for name in TXT_NAMES:
        fp = os.path.join(DATA_DIR, name)
        if exists(fp):
            d = read_vr_txt(fp)
            if d is not None and not d.empty:
                bundle[os.path.splitext(name)[0]] = d
    for fp in glob.glob(os.path.join(DATA_DIR, "*.txt")):
        base = os.path.basename(fp)
        if base in TXT_NAMES: continue
        if exists(fp):
            d = read_vr_txt(fp)
            if d is not None and not d.empty:
                bundle[os.path.splitext(base)[0]] = d
    return bundle

def assemble_table(bundle: dict) -> Optional[pd.DataFrame]:
    if not bundle: return None
    first_key = sorted(bundle.keys())[0]
    base = bundle[first_key].copy()
    base["timestamp"] = base["timestamp"].astype(float)
    wide = pd.DataFrame({"timestamp": base["timestamp"]})
    def role(k: str) -> str:
        return "eye" if "eye" in k.lower() else "head"
    for key, df in bundle.items():
        r = role(key)
        n = min(len(wide), len(df))
        if n == 0: continue
        wide.loc[:n-1, f"{r}_x"] = df["x"].values[:n]
        wide.loc[:n-1, f"{r}_y"] = df["y"].values[:n]
        wide.loc[:n-1, f"{r}_z"] = df["z"].values[:n]
    for c in ["head_x","head_y","head_z","eye_x","eye_y","eye_z"]:
        if c not in wide.columns: wide[c] = np.nan
    if any(k.lower().endswith("s") for k in bundle.keys()) and not any("ns" in k.lower() or "non" in k.lower() for k in bundle.keys()):
        lab = "Stroke"
    elif any("ns" in k.lower() or "non" in k.lower() for k in bundle.keys()) and not any(k.lower().endswith("s") for k in bundle.keys()):
        lab = "Non-Stroke"
    else:
        lab = "Unknown"
    wide["label"] = lab
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

def window_iter(df: pd.DataFrame, size:int=400, step:int=400):
    for start in range(0, len(df), step):
        chunk = df.iloc[start:start+size]
        if len(chunk) >= max(100, size//2):
            yield start, chunk

def plot_series(x, y, title):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title(title); ax.set_xlabel("Time (s)"); ax.set_ylabel("Value")
    st.pyplot(fig)

def align_to_model(model, feats_df: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "feature_names_in_"):
        cols = list(model.feature_names_in_)
        return feats_df.reindex(columns=cols, fill_value=0.0).to_numpy()
    return feats_df.select_dtypes(include=[np.number]).to_numpy()

def map_label(model, yhat):
    # prefer string classes if available
    if hasattr(model, "classes_"):
        classes = list(model.classes_)
        if set(classes) == {0,1}:
            return "Stroke" if int(yhat) == 1 else "Non-Stroke"
        # fallback for string labels
        return str(yhat)
    return "Stroke" if int(yhat) == 1 else "Non-Stroke"

# Load data (auto)
csv_df = load_first_csv()
if csv_df is not None:
    df_raw = csv_df
else:
    bundle = load_txt_bundle()
    df_raw = assemble_table(bundle) if bundle else pd.DataFrame(columns=["timestamp","head_x","head_y","head_z","eye_x","eye_y","eye_z","label"])

for c in ["head_x","head_y","head_z","eye_x","eye_y","eye_z"]:
    if c not in df_raw.columns: df_raw[c] = np.nan
if "timestamp" not in df_raw.columns:
    df_raw["timestamp"] = np.arange(len(df_raw))/60.0
if "label" not in df_raw.columns:
    df_raw["label"] = "Unknown"

# Try preload model
model = None
if exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception:
        model = None

# ========= Top: Simple Prediction =========
st.header("Quick Check")

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

btn_cols = st.columns([1,1,6])
with btn_cols[0]:
    b_demo = st.button("Predict from sliders")
with btn_cols[1]:
    b_data = st.button("Predict from data")

def predict_from_sliders():
    if model is None:
        st.write("Model not found.")
        return
    base = compute_features(df_raw)
    if hasattr(model, "feature_names_in_"):
        for c in model.feature_names_in_:
            if c not in base.columns:
                base[c] = 0.0
    demo = base.copy()
    def clamp(x, lo, hi): return max(lo, min(hi, x))
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

    X = align_to_model(model, demo)
    yhat = model.predict(X)[0]
    pred_label = map_label(model, yhat)
    st.subheader(f"Result: {pred_label}")
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        classes = list(getattr(model, "classes_", [0,1]))
        # pick index for "Stroke"
        if set(classes) == {0,1}:
            stroke_idx = 1
        else:
            stroke_idx = classes.index(next((c for c in classes if "stroke" in str(c).lower() and "non" not in str(c).lower()), classes[-1]))
        pct = float(proba[stroke_idx])
        st.progress(pct)
        st.write(f"Stroke probability: {pct*100:.1f}%")

def predict_from_data():
    if model is None:
        st.write("Model not found.")
        return
    feats = compute_features(df_raw)
    X = align_to_model(model, feats)
    yhat = model.predict(X)[0]
    pred_label = map_label(model, yhat)
    st.subheader(f"Result: {pred_label}")
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        classes = list(getattr(model, "classes_", [0,1]))
        if set(classes) == {0,1}:
            stroke_idx = 1
        else:
            stroke_idx = classes.index(next((c for c in classes if "stroke" in str(c).lower() and "non" not in str(c).lower()), classes[-1]))
        pct = float(proba[stroke_idx])
        st.progress(pct)
        st.write(f"Stroke probability: {pct*100:.1f}%")

if b_demo: predict_from_sliders()
if b_data: predict_from_data()

st.divider()

# ========= Tabs (simple wording) =========
tab1, tab2, tab3, tab4 = st.tabs(["Data Preview", "Charts", "Windows", "Train (optional)"])

with tab1:
    st.subheader("Data Preview")
    st.dataframe(df_raw.head(50), use_container_width=True)

with tab2:
    st.subheader("Charts")
    t = pd.to_numeric(df_raw["timestamp"], errors="coerce")
    for col in ["head_x","head_y","head_z","eye_x","eye_y","eye_z"]:
        if col in df_raw.columns and df_raw[col].notna().any():
            fig, ax = plt.subplots()
            ax.plot(t, pd.to_numeric(df_raw[col], errors="coerce"))
            ax.set_title(col)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Value")
            st.pyplot(fig)

with tab3:
    st.subheader("Window Summary")
    size = st.slider("Window size (samples)", 100, 3000, 500, step=50)
    step = st.slider("Step (samples)", 50, 3000, 500, step=50)
    feats, labels, starts = [], [], []
    for idx, chunk in window_iter(df_raw, size=size, step=step):
        f = compute_features(chunk)
        f["window_start"] = idx
        feats.append(f)
        labels.append(chunk["label"].mode().iloc[0] if "label" in chunk.columns and not chunk["label"].dropna().empty else "Unknown")
        starts.append(idx)
    if feats:
        feats_df = pd.concat(feats, ignore_index=True)
        feats_df["label"] = labels
        st.dataframe(feats_df.head(30), use_container_width=True)
    else:
        st.write("Not enough rows for the selected settings.")

with tab4:
    st.subheader("Train on This Data (optional)")
    if "feats_df" not in locals():
        feats_df = compute_features(df_raw)
        feats_df["label"] = df_raw["label"].mode().iloc[0] if "label" in df_raw.columns else "Unknown"
    X = feats_df.select_dtypes(include=[np.number]).copy()
    y = feats_df["label"] if "label" in feats_df.columns else pd.Series(["Unknown"]*len(X))
    if y.nunique() >= 2 and len(X) >= 6:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        clf.fit(Xtr, ytr)
        ypred = clf.predict(Xte)
        st.code(classification_report(yte, ypred), language="text")
        cm = confusion_matrix(yte, ypred, labels=sorted(y.unique()))
        fig, ax = plt.subplots()
        ax.imshow(cm)
        ax.set_xticks(range(len(sorted(y.unique())))); ax.set_yticks(range(len(sorted(y.unique()))))
        ax.set_xticklabels(sorted(y.unique()), rotation=45, ha="right"); ax.set_yticklabels(sorted(y.unique()))
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        st.pyplot(fig)
        if st.button("Save trained model"):
            os.makedirs("models", exist_ok=True)
            joblib.dump(clf, os.path.join("models","stroke_classifier.pkl"))
            st.write("Saved: models/stroke_classifier.pkl")
    else:
        st.write("Need at least two different labels to train.")
