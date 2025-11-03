# app.py — VR-Enabled Stroke Assessment Platform (Executive + Technical)
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

# ──────────────────────────────────────────────────────────────────────────────
# Page & constants
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="VR-Enabled Stroke Assessment", layout="wide")
st.title("🧠 VR-Enabled Stroke Assessment — Executive Overview & Technical Insights")
st.caption("Prototype demonstration for research/innovation discussions only. Not approved for clinical diagnosis or patient care.")

DATA_DIR = os.path.join(os.getcwd(), "data")
CSV_CANDIDATES = ["vr_feature_summary.csv", "vr_combined_raw.csv", "all_vr_data_raw.csv"]
DEFAULT_TXT_FILES = [
    "HeadPositionS.txt", "HeadRotationS.txt", "LeftEyeRotationS.txt", "RightEyeRotationS.txt",
    "HeadPositionNS.txt", "HeadRotationNS.txt", "LeftEyeRotationNS.txt", "RightEyeRotationNS.txt",
]
DEFAULT_MODEL_PATH = os.path.join("models", "stroke_classifier_from_raw.pkl")

# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────
def exists(path: str) -> bool:
    return os.path.exists(path) and (os.path.getsize(path) > 0)

def safe_json(obj):
    """Convert numpy / Pandas scalars to native Python types for Streamlit display."""
    if isinstance(obj, (np.generic, np.number)):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [safe_json(v) for v in obj]
    return obj

def load_first_existing_csv() -> Optional[pd.DataFrame]:
    """Prefer feature/combined CSVs if present in ./data."""
    for name in CSV_CANDIDATES:
        p = os.path.join(DATA_DIR, name)
        if exists(p):
            try:
                df = pd.read_csv(p)
                st.success(f"Loaded CSV: `{name}`  shape={df.shape}")
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
    """Parse text log lines like: 14:22:21.825, RecordLeftPoint, 0.18, 0.71, -0.05"""
    try:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 5:
            return None
        t = parts[0]  # HH:MM:SS(.ms) or other
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
        if df["timestamp_raw"].isna().all():
            df["timestamp"] = np.arange(len(df)) / 60.0
        else:
            first_valid = df["timestamp_raw"].dropna().iloc[0]
            df["timestamp"] = df["timestamp_raw"] - first_valid
        return df.drop(columns=["timestamp_raw"])
    except Exception as e:
        st.warning(f"Failed to parse {os.path.basename(filepath)}: {e}")
        return None

def load_repo_txt_bundle() -> dict:
    """Load expected .txt logs from ./data plus any other *.txt present."""
    bundle = {}
    if not os.path.isdir(DATA_DIR):
        return bundle
    for name in DEFAULT_TXT_FILES:
        fp = os.path.join(DATA_DIR, name)
        if exists(fp):
            df = read_vr_txt(fp)
            if df is not None and not df.empty:
                bundle[os.path.splitext(name)[0]] = df
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
    """
    if not bundle:
        return None
    first_key = sorted(bundle.keys())[0]
    base = bundle[first_key].copy()
    base = base.rename(columns={"x": f"{first_key}_x", "y": f"{first_key}_y", "z": f"{first_key}_z"})
    base["timestamp"] = base["timestamp"].astype(float)
    wide = base[["timestamp", f"{first_key}_x", f"{first_key}_y", f"{first_key}_z"]].copy()

    def role(name: str) -> str:
        n = name.lower()
        if "headposition" in n or "headrotation" in n:
            return "head"
        if "lefteyerotation" in n or "righteyerotation" in n or "eye" in n:
            return "eye"
        return "head"

    for key, df in bundle.items():
        if key == first_key:
            continue
        r = role(key)
        df2 = df.rename(columns={"x": f"{r}_x", "y": f"{r}_y", "z": f"{r}_z"}).copy()
        n = min(len(wide), len(df2))
        for col in [f"{r}_x", f"{r}_y", f"{r}_z"]:
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
    """Compute descriptive + simple velocity features from a sequence."""
    out = {}
    numeric_cols = ["head_x","head_y","head_z","eye_x","eye_y","eye_z"]
    for col in numeric_cols:
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            out[f"{col}_mean"] = s.mean()
            out[f"{col}_std"]  = s.std()
            out[f"{col}_mad"]  = (s - s.mean()).abs().mean()
            out[f"{col}_max"]  = s.max()
            out[f"{col}_min"]  = s.min()
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

def align_features_to_model(model, feats_df: pd.DataFrame) -> np.ndarray:
    """
    Align columns to the training schema of the model.
    - If model.feature_names_in_ exists, reindex to that exact order (fill missing with 0).
    - Else, fallback to numeric columns as-is.
    """
    if hasattr(model, "feature_names_in_"):
        want = list(model.feature_names_in_)
        aligned = feats_df.reindex(columns=want, fill_value=0.0)
        return aligned.to_numpy()
    return feats_df.select_dtypes(include=[np.number]).to_numpy()

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar: data source for technical tabs
# ──────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Data Source (for Technical Tabs)")
    source = st.radio("Choose source:", ["Auto-load from ./data", "Upload CSV", "Upload raw .txt bundle"], index=0)

df_raw = None
data_note = ""

if source == "Auto-load from ./data":
    df_csv = load_first_existing_csv()
    if df_csv is not None:
        df_raw = df_csv
        data_note = "Loaded from CSV in ./data"
    else:
        bundle = load_repo_txt_bundle()
        if bundle:
            df_raw = assemble_wide_table(bundle)
            data_note = "Assembled from TXT logs in ./data"
elif source == "Upload CSV":
    up = st.sidebar.file_uploader("Upload CSV (timestamp, head_x/y/z, eye_x/y/z, optional label)", type=["csv"])
    if up is not None:
        try:
            df_raw = pd.read_csv(up)
            data_note = "Uploaded CSV"
        except Exception as e:
            st.sidebar.error(f"Failed to read CSV: {e}")
elif source == "Upload raw .txt bundle":
    ups = st.sidebar.file_uploader("Upload 2–8 .txt logs (HeadPosition*, HeadRotation*, Left/RightEyeRotation*)",
                                   type=["txt","log"], accept_multiple_files=True)
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
            data_note = "Uploaded TXT bundle"

# Hygiene if df_raw exists
if df_raw is not None:
    for c in ["head_x","head_y","head_z","eye_x","eye_y","eye_z"]:
        if c not in df_raw.columns:
            df_raw[c] = np.nan
    if "timestamp" not in df_raw.columns:
        df_raw["timestamp"] = np.arange(len(df_raw)) / 60.0
    if "label" not in df_raw.columns:
        df_raw["label"] = "unknown"

# ──────────────────────────────────────────────────────────────────────────────
# QUICK PREDICTION (Executive-friendly) — shown FIRST
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("## 🎛️ Interactive Simulation: Rapid Assessment")
st.write("Adjust the parameters to simulate head–eye behavior. The pretrained model estimates the likelihood of **stroke** vs **non-stroke** patterns.")

# Load pretrained model for quick demo
clf_pretrained = None
if exists(DEFAULT_MODEL_PATH):
    try:
        clf_pretrained = joblib.load(DEFAULT_MODEL_PATH)
        st.success(f"Loaded pretrained model: `{DEFAULT_MODEL_PATH}`")
    except Exception as e:
        st.warning(f"Pretrained model found but failed to load: {e}")

c1, c2, c3, c4 = st.columns(4)
with c1:
    head_stability = st.slider("Head stability", 0.0, 1.0, 0.6, 0.01,
                               help="Higher = steadier head (lower variability)")
with c2:
    eye_fixation   = st.slider("Eye fixation", 0.0, 1.0, 0.6, 0.01,
                               help="Higher = steadier gaze (lower variability)")
with c3:
    head_range     = st.slider("Head movement range", 0.0, 2.0, 0.5, 0.01)
with c4:
    eye_range      = st.slider("Eye movement range", 0.0, 1.0, 0.2, 0.01)

if st.button("🧠 Predict from sliders", type="primary"):
    if clf_pretrained is None:
        st.warning("No pretrained model available. Add `models/stroke_classifier_from_raw.pkl` or upload a model in the tabs below.")
    else:
        # Build a 1-row feature DF using the model's expected columns (if known)
        if hasattr(clf_pretrained, "feature_names_in_"):
            cols = list(clf_pretrained.feature_names_in_)
        else:
            cols = [
                "head_x_mean","head_y_mean","head_z_mean",
                "eye_x_mean","eye_y_mean","eye_z_mean",
                "head_x_std","head_y_std","head_z_std",
                "eye_x_std","eye_y_std","eye_z_std",
                "head_x_max","head_y_max","head_z_max",
                "eye_x_max","eye_y_max","eye_z_max",
                "head_x_min","head_y_min","head_z_min",
                "eye_x_min","eye_y_min","eye_z_min",
            ]
        demo = pd.DataFrame(columns=cols); demo.loc[0] = 0.0

        # Heuristic mapping from sliders → features
        for name in demo.columns:
            nl = name.lower()
            if "eye" in nl and "std" in nl:
                demo.at[0, name] = max(1e-4, 0.5 * (1.0 - eye_fixation))
            elif "head" in nl and "std" in nl:
                demo.at[0, name] = max(1e-4, 0.5 * (1.0 - head_stability))
            elif "eye" in nl and ("range" in nl or "max" in nl):
                demo.at[0, name] = eye_range
            elif "head" in nl and ("range" in nl or "max" in nl):
                demo.at[0, name] = head_range
            elif "eye" in nl and "mean" in nl:
                demo.at[0, name] = 0.5 * eye_fixation
            elif "head" in nl and "mean" in nl:
                demo.at[0, name] = 0.5 * head_stability

        X_demo = align_features_to_model(clf_pretrained, demo)
        pred = clf_pretrained.predict(X_demo)[0]

        # Emphasized result card
        color = "#C1E1C1" if str(pred).lower().startswith("non") or str(pred) == "0" else "#F4CCCC"
        st.markdown(
            f"""
            <div style="background-color:{color};padding:1rem;border-radius:0.5rem;">
                <h4 style="margin:0;">🧠 Model-Estimated Classification: <b>{pred}</b></h4>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if hasattr(clf_pretrained, "predict_proba"):
            proba = clf_pretrained.predict_proba(X_demo)[0]
            probs = {str(k): float(v) for k, v in zip(getattr(clf_pretrained, "classes_", []), proba)}
            st.write("**Confidence distribution across outcome classes:**")
            st.json(safe_json(probs))

st.markdown("---")

# ──────────────────────────────────────────────────────────────────────────────
# TECHNICAL TABS
# ──────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["Data Preview", "Signals", "Windowed Features", "Train / Evaluate"])

with tab1:
    st.subheader("Data Preview")
    if df_raw is None:
        st.info("No data loaded yet. Use the sidebar to auto-load from `./data` or upload CSV/TXT.")
    else:
        st.success(f"Data ready: {data_note}. Shape: {df_raw.shape}")
        st.dataframe(df_raw.head(50))

with tab2:
    st.subheader("Time-Series Signals")
    if df_raw is None:
        st.info("Load data to view signals.")
    else:
        t = pd.to_numeric(df_raw["timestamp"], errors="coerce")
        for col in ["head_x","head_y","head_z","eye_x","eye_y","eye_z"]:
            if col in df_raw.columns and df_raw[col].notna().any():
                plot_series(t, pd.to_numeric(df_raw[col], errors="coerce"), f"{col} vs time")

with tab3:
    st.subheader("Windowed Feature Extraction")
    if df_raw is None:
        st.info("Load data to extract features.")
    else:
        size = st.slider("Window size (samples)", 100, 3000, 500, step=50)
        step = st.slider("Step size (samples)", 50, 3000, 500, step=50)
        feats, labels, starts = [], [], []
        for idx, chunk in window_iter(df_raw, size=size, step=step):
            f = compute_features(chunk); f["window_start_idx"] = idx
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
            st.session_state["feats_df"] = feats_df
        else:
            st.info("Increase the window or step to extract features.")

with tab4:
    st.subheader("Train / Evaluate (optional)")
    if df_raw is None:
        st.info("Load data to train/evaluate.")
    else:
        feats_df = st.session_state.get("feats_df")
        if feats_df is None:
            single = compute_features(df_raw)
            single["label"] = df_raw["label"].mode().iloc[0] if "label" in df_raw.columns else "unknown"
            feats_df = single

        featnum = feats_df.select_dtypes(include=[np.number]).copy()
        y = feats_df["label"] if "label" in feats_df.columns else pd.Series(["unknown"] * len(featnum))

        ok_classes = y.nunique(dropna=True) >= 2 and ("unknown" not in set(y.unique()) or y.nunique() > 2)
        if ok_classes and len(featnum) >= 6:
            X_train, X_test, y_train, y_test = train_test_split(
                featnum, y, test_size=0.30, random_state=42, stratify=y
            )
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
            st.info("Need at least **two label classes** across feature rows to train. "
                    "Tip: ensure both S and NS data are present, or use a mixed-label CSV.")
