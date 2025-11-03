# VR-Based Stroke Diagnosis System

A prototype diagnostic tool that uses **VR motion and eye-tracking data** to help distinguish **stroke vs. non-stroke** cases in emergency department settings.

---

## Problem

Diagnosing dizziness-related stroke in the ER is challenging - traditional assessments are subjective and time-consuming. This project introduces a **VR-based system** that records patient head and eye movement to assist clinicians through objective motion data analysis.

---

## Solution

Using a **VR headset**, patients perform guided visual tasks while their **head and eye rotation data** are captured in real-time.  
These signals are analyzed through a **machine learning classifier** trained to detect subtle differences between stroke and non-stroke cases.

---

## Data Overview

- `HeadPosition.txt`, `HeadRotation.txt`, `LeftEyeRotation.txt`, `RightEyeRotation.txt`
- Stroke (`S`) and Non-Stroke (`NS`) session data
- Processed feature summary: `vr_feature_summary.csv`

Each dataset contains:
timestamp, x, y, z
representing spatial and rotational movement patterns during VR tasks.

---

## Model Pipeline

1. **Data Preprocessing** – Cleaning and merging raw VR sensor logs  
2. **Feature Extraction** – Mean, variance, angular velocity, smoothness  
3. **Model Training** – Random Forest Classifier (0.89 accuracy)  
4. **Evaluation** – Confusion matrix, feature importance visualization  

Code files:
- `strokesl.py` - main ML pipeline  
- `stroke_analysis.ipynb` - EDA, visualization, and model validation

---

## Tech Stack

| Category | Tools |
|-----------|-------|
| Data Processing | Python, Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML & Modeling | scikit-learn |
| Environment | Jupyter Notebook |

---

## Outcome

The prototype demonstrates the **potential of motion-based diagnostics** for early stroke detection and could be extended for remote neurological assessment.
Streamlit Demo: https://vr-stroke-diagnosis.streamlit.app/

---
