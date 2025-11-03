"""
VR Stroke Screening App (Improved)

This Streamlit application demonstrates a simple, user‑friendly interface for exploring
head and eye tracking data and predicting whether a motion pattern is more
consistent with a stroke or non‑stroke subject.  Unlike earlier versions, this
app automatically adapts to the columns present in your data set, plots all
available numeric signals, and produces a clear classification result in plain
English.  It is designed to run out of the box with the data and model
distributed in this repository.

Highlights
----------
* Automatically loads the first available CSV (e.g. ``vr_combined_raw.csv`` or
  ``vr_feature_summary.csv``) or assembles data from the raw ``.txt`` logs.
* Detects which numeric columns are available and builds charts for each one.
* Supports both ``x``, ``y`` and ``z`` style models and more complex feature
  sets; when the model expects ``x``, ``y`` and ``z`` the sliders adjust those
  three values directly.
* Presents predictions as "Stroke" or "Non‑Stroke" with a probability bar.

To run this app locally:

.. code-block:: bash

    pip install streamlit pandas numpy matplotlib scikit-learn joblib
    streamlit run app_improved.py

"""

import os
import glob
import io
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DATA_DIRS = ["data", "."]
CSV_CANDIDATES = [
    "vr_combined_raw.csv",
    "vr_feature_summary.csv",
    "all_vr_data_raw.csv",
]
TXT_NAMES = [
    "HeadPositionS.txt",
    "HeadRotationS.txt",
    "LeftEyeRotationS.txt",
    "RightEyeRotationS.txt",
    "HeadPositionNS.txt",
    "HeadRotationNS.txt",
    "LeftEyeRotationNS.txt",
    "RightEyeRotationNS.txt",
]
MODEL_CANDIDATES = [
    os.path.join("models", "stroke_classifier_from_raw.pkl"),
    "stroke_classifier_from_raw.pkl",
]


def file_exists(path: str) -> bool:
    """Return True if the file exists and is non‑empty."""
    return os.path.exists(path) and os.path.getsize(path) > 0


def load_first_available_csv() -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Return the first CSV found in the candidate list along with its path."""
    for base in DATA_DIRS:
        for name in CSV_CANDIDATES:
            path = os.path.join(base, name)
            if file_exists(path):
                try:
                    return pd.read_csv(path), path
                except Exception:
                    # Skip files that cannot be parsed
                    continue
    return None, None


def parse_log_line(line: str) -> Optional[Tuple[float, float, float, float]]:
    """Parse a single line from a VR log file into a timestamp and xyz tuple.

    Lines are expected to have the form ``HH:MM:SS.sss, RecordType, x, y, z``.
    The timestamp is converted to seconds.  If parsing fails, None is returned.
    """
    parts = [p.strip() for p in line.split(",")]
    if len(parts) < 5:
        return None
    try:
        # Convert HH:MM:SS.sss to seconds
        if ":" in parts[0]:
            hh, mm, ss = parts[0].split(":")
            timestamp = int(hh) * 3600 + int(mm) * 60 + float(ss)
        else:
            timestamp = float(parts[0])
        x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
        return timestamp, x, y, z
    except Exception:
        return None


def read_vr_log(path: str) -> Optional[pd.DataFrame]:
    """Read a VR log file into a DataFrame with columns ``timestamp``, ``x``, ``y``, ``z``."""
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            rec = parse_log_line(line)
            if rec:
                rows.append(rec)
    if not rows:
        return None
    df = pd.DataFrame(rows, columns=["timestamp", "x", "y", "z"])
    # Normalize timestamps to start at zero
    if df["timestamp"].notna().any():
        df["timestamp"] = df["timestamp"] - df["timestamp"].iloc[0]
    return df


def load_logs_to_table() -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Assemble a DataFrame from any available raw ``.txt`` logs.

    The resulting table contains ``timestamp``, ``head_x``, ``head_y``, ``head_z``,
    ``eye_x``, ``eye_y`` and ``eye_z`` if possible.  If no logs are present
    returns (None, None).
    """
    bundle = {}
    for base in DATA_DIRS:
        if not os.path.isdir(base):
            continue
        # Load expected names first
        for name in TXT_NAMES:
            path = os.path.join(base, name)
            if file_exists(path):
                df = read_vr_log(path)
                if df is not None and not df.empty:
                    bundle[os.path.splitext(name)[0]] = df
        # Load any remaining txt logs
        for path in glob.glob(os.path.join(base, "*.txt")):
            base_name = os.path.basename(path)
            if base_name in TXT_NAMES:
                continue
            if file_exists(path):
                df = read_vr_log(path)
                if df is not None and not df.empty:
                    bundle[os.path.splitext(base_name)[0]] = df
    if not bundle:
        return None, None

    # Merge logs on their indices; this is a simple alignment strategy that
    # preserves the original sample order while combining multiple series.  We
    # label each series as either ``head`` or ``eye`` based on its filename.
    def role(name: str) -> str:
        return "eye" if "eye" in name.lower() else "head"

    first_key = sorted(bundle.keys())[0]
    base_df = bundle[first_key].copy().rename(columns={"x": f"{role(first_key)}_x",
                                                       "y": f"{role(first_key)}_y",
                                                       "z": f"{role(first_key)}_z"})
    result = pd.DataFrame({"timestamp": base_df["timestamp"]})
    result[[f"{role(first_key)}_x", f"{role(first_key)}_y", f"{role(first_key)}_z"]] = base_df[[f"{role(first_key)}_x",
                                                                                                   f"{role(first_key)}_y",
                                                                                                   f"{role(first_key)}_z"]]
    # Merge remaining logs
    for key, df in bundle.items():
        if key == first_key:
            continue
        r = role(key)
        temp = df.copy().rename(columns={"x": f"{r}_x", "y": f"{r}_y", "z": f"{r}_z"})
        # Align on sample count (outer join by position)
        n = min(len(result), len(temp))
        for col in [f"{r}_x", f"{r}_y", f"{r}_z"]:
            if col in temp.columns:
                result.loc[:n-1, col] = temp[col].values[:n]
    return result, "assembled from logs"


def load_data() -> Tuple[pd.DataFrame, str]:
    """Load motion data from CSV or logs.

    The function tries to load a CSV first; if none are found it will attempt to
    assemble a table from any available ``.txt`` logs.  Returns the DataFrame
    and a string describing the source.
    """
    csv_df, csv_path = load_first_available_csv()
    if csv_df is not None:
        return csv_df, f"loaded from {os.path.basename(csv_path)}"
    log_df, origin = load_logs_to_table()
    if log_df is not None:
        return log_df, origin
    # No data available; return empty frame
    empty_df = pd.DataFrame(columns=["timestamp", "x", "y", "z", "label"])
    return empty_df, "no data found"


def load_model() -> Tuple[RandomForestClassifier, str]:
    """Load the first available classification model and return it with a path."""
    for path in MODEL_CANDIDATES:
        if file_exists(path):
            try:
                model = joblib.load(path)
                return model, path
            except Exception:
                continue
    raise FileNotFoundError("No pretrained model file was found. Ensure that "
                            "'stroke_classifier_from_raw.pkl' is present in the 'models' folder.")


def main() -> None:
    # Page title and description
    st.header("Quick Motion Check")
    st.write(
        "This tool uses motion and eye‑tracking data collected in a VR headset "
        "to estimate whether a subject exhibits patterns consistent with a stroke."
    )

    # Load model and data
    try:
        model, model_path = load_model()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

    df_raw, source_desc = load_data()
    if df_raw.empty:
        st.warning("No data available. Please add CSV or log files to the 'data' folder.")

    # Identify numeric columns for plotting and input
    numeric_cols = [c for c in df_raw.columns if df_raw[c].dtype != object and c.lower() not in {"label"}]

    # Determine whether the model expects simple xyz inputs or more complex features
    expected = list(getattr(model, "feature_names_in_", []))
    simple_mode = False
    if expected:
        # If the model was trained on a handful of axes (x,y,z or similar)
        cleaned = [c.lower().replace("axis", "").lstrip("_") for c in expected]
        if len(cleaned) <= 3 and all(ch in {"x", "y", "z"} for ch in cleaned):
            simple_mode = True
    else:
        # Fallback: if the CSV clearly has x,y,z
        if set(df_raw.columns[:3]).issuperset({"x", "y", "z"}):
            simple_mode = True

    # Quick prediction interface
    if simple_mode:
        st.subheader("Select axis values")
        # Use ranges from data if available
        def get_range(col: str, default_min: float, default_max: float) -> Tuple[float, float]:
            if col in df_raw.columns and df_raw[col].notna().any():
                # Use 1st and 99th percentiles to avoid outliers
                q1, q99 = np.nanpercentile(df_raw[col], [1, 99])
                if q1 < q99:
                    return float(q1), float(q99)
            return default_min, default_max

        x_min, x_max = get_range("x", -20.0, 20.0)
        y_min, y_max = get_range("y", -50.0, 150.0)
        z_min, z_max = get_range("z", -20.0, 20.0)

        col1, col2, col3 = st.columns(3)
        x_val = col1.slider("X‑axis motion", x_min, x_max, (x_min + x_max) / 2.0)
        y_val = col2.slider("Y‑axis motion", y_min, y_max, (y_min + y_max) / 2.0)
        z_val = col3.slider("Z‑axis motion", z_min, z_max, (z_min + z_max) / 2.0)

        if st.button("Predict"):  # Only compute when button clicked
            # Build input row based on expected order
            if expected:
                row = []
                for name in expected:
                    name_l = name.lower().replace("axis", "").lstrip("_")
                    if name_l.startswith("x"):
                        row.append(x_val)
                    elif name_l.startswith("y"):
                        row.append(y_val)
                    elif name_l.startswith("z"):
                        row.append(z_val)
                    else:
                        row.append(0.0)
            else:
                # Default order x,y,z
                row = [x_val, y_val, z_val]
            X = np.array([row], dtype=float)
            yhat = model.predict(X)[0]
            label = map_label(model, yhat)
            st.subheader(f"Result: {label}")
            # Probability bar
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                classes = list(getattr(model, "classes_", [0, 1]))
                # Determine index of stroke class
                stroke_idx = 1 if set(classes) == {0, 1} else classes.index(
                    next((c for c in classes if "stroke" in str(c).lower() and "non" not in str(c).lower()), classes[-1])
                )
                pct = float(proba[stroke_idx])
                st.progress(pct)
                st.write(f"Stroke probability: {pct * 100:.1f}%")
    else:
        # Complex mode: adjust features indirectly
        st.subheader("Adjust motion characteristics")
        col1, col2 = st.columns(2)
        head_stability = col1.slider("Head stability", 0.0, 1.0, 0.7, 0.01)
        eye_stability = col2.slider("Eye stability", 0.0, 1.0, 0.7, 0.01)
        range_factor = st.slider("Movement range factor", 0.5, 2.0, 1.0, 0.01)
        # Compute base features from current data as starting point
        base = compute_features(df_raw)
        # Ensure all expected columns exist
        for c in expected:
            if c not in base.columns:
                base[c] = 0.0
        # Adjust features heuristically
        demo = base.copy()
        for col in demo.columns:
            lc = col.lower()
            if "std" in lc or "vel_std" in lc:
                if "head" in lc:
                    demo[col] = demo[col].astype(float) * (1.0 + (1.0 - head_stability))
                if "eye" in lc:
                    demo[col] = demo[col].astype(float) * (1.0 + (1.0 - eye_stability))
            if "mean" in lc:
                if "head" in lc:
                    demo[col] = demo[col].astype(float) + (0.5 - head_stability)
                if "eye" in lc:
                    demo[col] = demo[col].astype(float) + (0.5 - eye_stability)
            if "max" in lc or "min" in lc or "range" in lc:
                demo[col] = demo[col].astype(float) * range_factor
        # Predict on click
        if st.button("Predict"):
            X = demo[expected].to_numpy() if expected else demo.select_dtypes(include=[np.number]).to_numpy()
            yhat = model.predict(X)[0]
            label = map_label(model, yhat)
            st.subheader(f"Result: {label}")
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X)[0]
                classes = list(getattr(model, "classes_", [0, 1]))
                stroke_idx = 1 if set(classes) == {0, 1} else classes.index(
                    next((c for c in classes if "stroke" in str(c).lower() and "non" not in str(c).lower()), classes[-1])
                )
                pct = float(proba[stroke_idx])
                st.progress(pct)
                st.write(f"Stroke probability: {pct * 100:.1f}%")

    st.markdown("---")

    # Data exploration tabs
    tabs = st.tabs(["Data", "Charts", "Windows"])
    # Tab 1: Data preview
    with tabs[0]:
        st.subheader("Data Preview")
        st.write(f"Data source: {source_desc}")
        st.dataframe(df_raw.head(50))

    # Tab 2: Charts
    with tabs[1]:
        st.subheader("Time Series Charts")
        if not numeric_cols:
            st.write("No numeric columns available for plotting.")
        else:
            time_col = "timestamp" if "timestamp" in df_raw.columns else None
            x_axis = pd.to_numeric(df_raw[time_col], errors="coerce") if time_col else np.arange(len(df_raw))
            for col in numeric_cols:
                st.write(f"**{col}**")
                fig, ax = plt.subplots()
                ax.plot(x_axis, pd.to_numeric(df_raw[col], errors="coerce"))
                ax.set_xlabel(time_col.capitalize() if time_col else "Sample index")
                ax.set_ylabel(col)
                st.pyplot(fig)

    # Tab 3: Window summary (optional)
    with tabs[2]:
        st.subheader("Window Summary")
        window_size = st.slider("Window size (samples)", 100, 2000, 500, 50)
        step_size = st.slider("Step (samples)", 50, 2000, 500, 50)
        features_list = []
        labels_list = []
        for start in range(0, len(df_raw), step_size):
            end = start + window_size
            chunk = df_raw.iloc[start:end]
            if len(chunk) < window_size / 2:
                continue
            feats = compute_features(chunk)
            feats["start_index"] = start
            features_list.append(feats)
            labels_list.append(chunk.get("label", pd.Series(["Unknown"])).mode().iloc[0])
        if features_list:
            feat_df = pd.concat(features_list, ignore_index=True)
            feat_df["label"] = labels_list
            st.dataframe(feat_df.head(30))
        else:
            st.write("No windows available for the selected parameters.")


def map_label(model, yhat):
    """Map a raw classifier output to a human‑readable label.

    If the model uses numeric labels 0/1, those are mapped to
    ``Non‑Stroke`` and ``Stroke`` respectively.  Otherwise, the raw label is
    returned as a string.
    """
    try:
        # If model.classes_ is numeric, map 1->Stroke
        classes = getattr(model, "classes_", [0, 1])
        if set(classes) == {0, 1}:
            return "Stroke" if int(yhat) == 1 else "Non‑Stroke"
        return str(yhat)
    except Exception:
        return str(yhat)


if __name__ == "__main__":
    main()
