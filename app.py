# app.py — VR Stroke Diagnosis Demo (uses your real repo data + pretrained model)
# Run: streamlit run app.py

import os
import io
import glob
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from typing import Optional, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="VR Stroke Diagnosis — Motion & Eye Tracking", layout="wide")
st.title("🧠 VR-Based Stroke Diagnosis — Motion & Eye Tracking")

st.markdown(
    "This app loads **head & eye-tracking signals** from your repository’s `./data/` folder, "
    "extracts features, and (a) **auto-loads your pretrained model** for instant prediction, or "
    "(b) lets you train/evaluate a simple classifier."
)

# -------------------------
# Repo-relative data paths
# -------------------------
DATA_DIR = os.path.join(os.getcwd(), "data")
DEFAULT_TXT_FILES = [
    "HeadPositionS.txt", "HeadRotationS.txt", "LeftEyeRotationS.txt", "RightEyeRotationS.txt",
    "HeadPositionNS.txt", "HeadRotationNS.txt", "LeftEyeRotationNS.txt", "RightEyeRotationNS.txt",
]
CSV_CANDIDATES = ["vr_feature_summary.csv", "vr_combined_raw.csv", "all_vr_data_raw.csv"]
DEFAULT_MODEL_PATH = os.path.join("models", "stroke_classifier_from_raw.pkl")

# -------------------------
# Helpers
# -------------------------
def exists(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0

def load_first_existing_csv() -> Optional[pd.DataFrame]:
    for name in CSV_CANDIDATES:
        p = os.path.join(DATA_DIR, name)
        if exists(p):
            try:
                df = pd.read_csv(p)
                st.success(f"Loaded CSV: `{name}` with shape {df.shape}")
                return df
            except Exception as e:
                st.warning(f"Found `{name}` but failed to read: {e}")
    return None

def detect_label_from_filename(fname: str) -> Optional[str]:
    f = fname.lower()
    if "ns" in f or "non" in f:
        return "non_stroke"
    if f.endswith("s.txt") or "_s" in f or "stroke" in f:
        return "stroke"
    return None

def parse_record_line(line: str) -> Optional[Tuple[float, float, float, float]]:
    """
    Parse a line like:
      14:22:21.825, RecordLeftPoint, 0.1839331, 0.7152903, -0.05265599
    Returns: (sec_float, x, y, z) or None
    """
    try:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            return None
        t = parts[0]  # time token
        sec = np.nan
        if ":" in t:
            hh, mm, ss = t.split(":")
            sec = int(hh) * 3600 + int(mm) * 60 + float(ss)
        x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
        return sec, x, y, z
    except Exception:
        return None

def read_vr_txt(filepath: str) -> Optional[pd.DataFrame]:
    rows = []
    try:
        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = parse_record_line(line)
                if rec is None:
                    continue
                rows.append(rec)
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=["timestamp_raw", "x", "y", "z"])
        # build a monotonic timestamp if needed
        if df["timestamp_raw"].isna().all():
            df["timestamp"] = np.arange(len(df)) / 60.0
        else:
            first_valid = df["timestamp_raw"].dropna().iloc[0]
            df["timestamp"] = df["timestamp_raw"] - first_valid
        df = df.drop(columns=["timestamp_raw"])
        return df
    except Exception as e:
        st.warning(f"Failed to parse {os.path.basename(filepath)}: {e}")
        return None

def load_repo_txt_bundle() -> dict:
    bundle = {}
    if not os.path.isdir(DATA_DIR):
        return bundle
    # expected names first
    for name in DEFAULT_TXT_FILES:
        fp = os.path.join(DATA_DIR, name)
        if exists(fp):
            df = read_vr_txt(fp)
            if df is not None and not df.empty:
                bundle[os.path.splitext(name)[0]] = df
    # any other .txt in data/
    for fp in glob.glob(os.path.join(DATA_DIR, "*.txt")):
        base = os.path.basename(fp)
        if base in DEFAULT_TXT_FILES:
            continue
        if exists(fp):
            df = read_vr_txt(fp)
            if df is not None and not df.empty:
                bundle[os.path.splitext(base)[0]] = df
    return bundle

def assemble_wide_table(bundle: dict) -> Optional[pd.DataFrame]:
    """
    Merge available series into a single wide table:
    timestamp, head_x, head_y, head_z, eye_x, eye_y, eye_z, label
    Label inferred from filenames (S/NS).
    """
    if not bundle:
        return None

    first_key = sorted(bundle.keys())[0]
    base = bundle[first_key].copy()
    base = base.rename(columns={"x": f"{first_key}_x", "y": f"{first_key}_y", "z": f"{first_key}_z"})
    base["timestamp"] = base["timestamp"].astype(float)
    wide = base[["timestamp", f"{first_key}_x", f"{first_key}_y", f"{first_key}_z"]].copy()

    def map_role(name: str) -> str:
        n = name.lower()
        if "headposition" in n or "headrotation" in n:
            return "head"
        if "lefteyerotation" in n or "righteyerotation" in n or "eye" in n:
            return "eye"
        return "head"

    for key, df in bundle.items():
        if key == first_key:
            continue
        df2 = df.copy()
        role = map_role(key)
        df2 = df2.rename(columns={"x": f"{role}_x", "y": f"{role}_y", "z": f"{role}_z"})
        n = min(len(wide), len(df2))
        for col in [f"{role}_x", f"{role}_y", f"{role}_z"]:
            if col in df2.columns:
                wide.loc[:n-1, col] = df2[col].values[:n]

    # infer single bundle label from filenames present
    labels = [detect_label_from_filename(k) or "unknown" for k in bundle.keys()]
    if "stroke" in labels and "non_stroke" not in labels:
        final_label = "stroke"
    elif "non_stroke" in labels and "stroke" not in labels:
        final_label = "non_stroke"
    else:
        final_label = "unknown"

    wide["label"] = final_label
    for c in ["head_x","head_y","head_z","eye_x","eye_y","eye_z"]:
        if c not in wide.columns:
            wide[c] = np.nan
    return wide[["timestamp","head_x","head_y","head_z","eye_x","eye_y","eye_z","label"]]

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive + simple velocity features."""
    out = {}
    numeric_cols = ["head_x","head_y","head_z","eye_x","eye_y","eye_z"]
    for col in numeric_cols:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            out[f"{col}_mean"] = s.mean()
            out[f"{col}_std"] = s.std()
            out[f"{col}_mad"] = (s - s.mean()).abs().mean()
            out[f"{col}_max"] = s.max()
            out[f"{col}_min"] = s.min()
    if "timestamp" in df.columns:
        t = pd.to_numeric(df["timestamp"], errors="coerce").values
        for col in numeric_cols:
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").values
                dv = np.diff(vals)
                dt = np.diff(t[:len(vals)])
                with np.errstate(divide="ignore", invalid="ignore"):
                    vel = np.where(dt != 0, dv / dt, np.nan)
                if vel.size > 1:
                    out[f"{col}_vel_mean"] = np.nanmean(vel)
                    out[f"{col}_vel_std"]  = np.nanstd(vel)
    return pd.DataFrame([out])

def window_iter(df: pd.DataFrame, size:int=400, step:int=400):
    for start in range(0, len(df), step):
        chunk = df.iloc[start:start+size]
        if len(chunk) >= max(100, size//2):
            yield start, chunk

def plot_series(x, y, title, xlabel="timestamp (s)", ylabel="value"):
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    st.pyplot(fig)

# -------------------------
# Data loading
# -------------------------
st.sidebar.header("Data Source")
source = st.sidebar.radio(
    "Choose source:",
    ["Auto-load from ./data", "Upload CSV", "Upload raw .txt bundle"],
    index=0
)

df_raw = None
note = ""

if source == "Auto-load from ./data":
    df_csv = load_first_existing_csv()
    if df_csv is not None:
        df_raw = df_csv
        note = "Loaded from CSV in ./data"
    else:
        bundle = load_repo_txt_bundle()
        if bundle:
            df_raw = assemble_wide_table(bundle)
            note = "Assembled from TXT logs in ./data"
        else:
            st.info("No CSV or expected .txt logs found in ./data. Try uploading files.")
elif source == "Upload CSV":
    up = st.sidebar.file_uploader("Upload CSV (columns: timestamp, head_x, head_y, head_z, eye_x, eye_y, eye_z, optional label)", type=["csv"])
    if up is not None:
        try:
            df_raw = pd.read_csv(up)
            note = "Uploaded CSV"
        except Exception as e:
            st.error(f"Failed to read CSV: {e}")
elif source == "Upload raw .txt bundle":
    st.sidebar.write("Upload 2–8 `.txt` logs (e.g., HeadPositionS.txt, HeadRotationNS.txt, LeftEyeRotationS.txt, ...)")
    ups = st.sidebar.file_uploader("Upload .txt logs", type=["txt","log"], accept_multiple_files=True)
    if ups:
        bundle = {}
        for f in ups:
            buf = io.StringIO(f.getvalue().decode("utf-8", errors="ignore"))
            rows = []
            for line in buf:
                rec = parse_record_line(line)
                if rec is None: 
                    continue
                rows.append(rec)
            if rows:
                df = pd.DataFrame(rows, columns=["timestamp_raw","x","y","z"])
                if df["timestamp_raw"].isna().all():
                    df["timestamp"] = np.arange(len(df)) / 60.0
                else:
                    first_valid = df["timestamp_raw"].dropna().iloc[0]
                    df["timestamp"] = df["timestamp_raw"] - first_valid
                df = df.drop(columns=["timestamp_raw"])
                bundle[os.path.splitext(f.name)[0]] = df
        if bundle:
            df_raw = assemble_wide_table(bundle)
            note = "Uploaded TXT bundle"

if df_raw is None:
    st.stop()

# Basic hygiene
for c in ["head_x","head_y","head_z","eye_x","eye_y","eye_z"]:
    if c not in df_raw.columns:
        df_raw[c] = np.nan
if "timestamp" not in df_raw.columns:
    df_raw["timestamp"] = np.arange(len(df_raw)) / 60.0
if "label" not in df_raw.columns:
    df_raw["label"] = "unknown"

st.success(f"Data ready: {note}. Shape: {df_raw.shape}")

# -------------------------
# Preview & Signals
# -------------------------
tab1, tab2, tab3 = st.tabs(["Preview", "Signals", "Windowed Features"])

with tab1:
    st.subheader("Dataset Preview")
    st.dataframe(df_raw.head(50))

with tab2:
    st.subheader("Time-Series Signals")
    t = pd.to_numeric(df_raw["timestamp"], errors="coerce")
    for col in ["head_x","head_y","head_z","eye_x","eye_y","eye_z"]:
        if col in df_raw.columns and df_raw[col].notna().any():
            plot_series(t, pd.to_numeric(df_raw[col], errors="coerce"), f"{col} vs time")

with tab3:
    st.subheader("Windowed Feature Extraction")
    size = st.slider("Window size (samples)", 100, 3000, 500, step=50)
    step = st.slider("Step size (samples)", 50, 3000, 500, step=50)
    feats, labels, starts = [], [], []
    for idx, chunk in window_iter(df_raw, size=size, step=step):
        f = compute_features(chunk)
        f["window_start_idx"] = idx
        feats.append(f)
        if "label" in chunk.columns and not chunk["label"].dropna().empty:
            labels.append(chunk["label"].mode().iloc[0])
        else:
            labels.append("unknown")
        starts.append(idx)
    if feats:
        feats_df = pd.concat(feats, ignore_index=True)
        feats_df["label"] = labels
        st.write("Feature table (first 30 rows):")
        st.dataframe(feats_df.head(30))
    else:
        st.info("Increase the window or step to extract features.")

# -------------------------
# Modeling
# -------------------------
st.header("🔎 Prediction & Training")

left, right = st.columns(2)

# --- Right: Inference first (auto-load your pretrained model) ---
with right:
    st.subheader("Inference (pretrained or uploaded model)")
    clf_pretrained = None
    if exists(DEFAULT_MODEL_PATH):
        try:
            clf_pretrained = joblib.load(DEFAULT_MODEL_PATH)
            st.success(f"✅ Loaded pretrained model: `{DEFAULT_MODEL_PATH}`")
            f_inf = compute_features(df_raw).select_dtypes(include=[np.number])
            pred = clf_pretrained.predict(f_inf)[0]
            st.markdown(f"### 🧠 Predicted condition: `{pred}`")
            if hasattr(clf_pretrained, "predict_proba"):
                proba = clf_pretrained.predict_proba(f_inf)[0]
                st.write("Class probabilities:", dict(zip(getattr(clf_pretrained, "classes_", []), map(float, proba))))
        except Exception as e:
            st.warning(f"Pretrained model found but failed to load: {e}")

    up_model = st.file_uploader("Or upload a different model (.pkl)", type=["pkl"])
    if up_model is not None:
        try:
            clf2 = joblib.load(up_model)
            st.success("Model loaded.")
            f_inf = compute_features(df_raw).select_dtypes(include=[np.number])
            pred = clf2.predict(f_inf)[0]
            st.markdown(f"**Prediction:** `{pred}`")
            if hasattr(clf2, "predict_proba"):
                proba = clf2.predict_proba(f_inf)[0]
                st.write("Class probabilities:", dict(zip(getattr(clf2, "classes_", []), map(float, proba))))
        except Exception as e:
            st.error(f"Failed to use uploaded model: {e}")

# --- Left: Optional training for transparency ---
with left:
    st.subheader("Optional: Train from current data")
    # If we created feats_df in the features tab, reuse it; else compute a single-window feature row
    if "feats_df" not in locals():
        if "feats_df" in globals():
            feats_df = globals()["feats_df"]
        else:
            single = compute_features(df_raw)
            single["label"] = df_raw["label"].mode().iloc[0] if "label" in df_raw.columns else "unknown"
            feats_df = single

    featnum = feats_df.select_dtypes(include=[np.number]).copy()
    y = feats_df["label"] if "label" in feats_df.columns else pd.Series(["unknown"]*len(featnum))

    ok_classes = y.nunique(dropna=True) >= 2 and ("unknown" not in set(y.unique()) or y.nunique() > 2)
    if ok_classes and len(featnum) >= 6:
        X_train, X_test, y_train, y_test = train_test_split(featnum, y, test_size=0.3, random_state=42, stratify=y)
        clf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        st.code(classification_report(y_test, y_pred), language="text")
        cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
        fig, ax = plt.subplots()
        ax.imshow(cm)
        ax.set_xticks(range(len(sorted(y.unique()))))
        ax.set_yticks(range(len(sorted(y.unique()))))
        ax.set_xticklabels(sorted(y.unique()), rotation=45, ha="right")
        ax.set_yticklabels(sorted(y.unique()))
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        st.pyplot(fig)

        importances = pd.Series(clf.feature_importances_, index=featnum.columns).sort_values(ascending=False).head(20)
        fig2, ax2 = plt.subplots()
        ax2.barh(importances.index[::-1], importances.values[::-1])
        ax2.set_title("Top Feature Importances")
        st.pyplot(fig2)

        if st.button("💾 Save trained model (models/stroke_classifier.pkl)"):
            os.makedirs("models", exist_ok=True)
            joblib.dump(clf, "models/stroke_classifier.pkl")
            st.success("Saved: models/stroke_classifier.pkl")
    else:
        st.info("Need at least **two label classes** across feature rows to train.\n"
                "Tip: ensure both S and NS data are present or use a mixed-label CSV.")

st.markdown("---")
st.caption("Portfolio demo. Not for clinical use. Additional validation & regulatory approvals are required for real-world deployment.")
