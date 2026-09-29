# -*- coding: utf-8 -*-
"""
CKM syndrome incident MACE risk prediction — parsimonious 6-feature model.
Streamlit app. Run:  streamlit run app.py
"""
import os
import pickle

import numpy as np
import streamlit as st

st.set_page_config(page_title="CKM-MACE Risk Prediction", page_icon="🫀", layout="centered")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

FEATURE_LABELS = {
    "age": "Age (years)",
    "neut_pct": "Neutrophil percentage (%)",
    "rbc": "Red blood cell count (×10¹²/L)",
    "crea": "Creatinine (µmol/L)",
    "egfr": "eGFR (mL/min/1.73 m²)",
    "n_comorb": "Comorbidity count (0–10)",
}
FEATURES = ["age", "neut_pct", "rbc", "crea", "egfr", "n_comorb"]
DEFAULTS = {"age": 60, "neut_pct": 65.0, "rbc": 4.0, "crea": 100.0, "egfr": 60.0, "n_comorb": 2}
MINMAX = {
    "age": (18, 100), "neut_pct": (0.0, 100.0), "rbc": (1.0, 8.0),
    "crea": (20.0, 1500.0), "egfr": (5.0, 150.0), "n_comorb": (0, 10),
}

@st.cache_resource
def load_artifacts():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

art = load_artifacts()
model = art["model"]
scaler = art["scaler"]

st.title("🫀 CKM Syndrome MACE Risk Prediction")
st.markdown(
    "A **parsimonious 6-feature machine learning model** developed on a three-center CKD cohort, "
    "predicting the risk of incident major adverse cardiovascular events (MACE: heart failure / "
    "myocardial infarction / stroke)."
)

with st.sidebar:
    st.header("Model information")
    st.markdown(
        """
        - **Model**: logistic regression (LASSO penalty, C=0.01)
        - **Features**: 6 Boruta-selected stable predictors
        - **Internal AUC**: 0.768 (5-fold cross-validation)
        - **External validation**: 0.752 (Provincial) / 0.808 (Central, non-laboratory subset)
        - **Calibration Brier**: 0.094 (full model)
        - **Derivation cohort**: Qianfoshan Hospital, 3,225 patients (MACE 389, 12.1%)
        """
    )
    st.caption("Intended for the cardio-kidney-metabolic (CKM) syndrome population")

st.subheader("Enter patient characteristics")
col1, col2 = st.columns(2)
inputs = {}
for i, feat in enumerate(FEATURES):
    col = col1 if i % 2 == 0 else col2
    lo, hi = MINMAX[feat]
    if isinstance(lo, float):
        inputs[feat] = col.number_input(FEATURE_LABELS[feat], lo, hi, DEFAULTS[feat], step=0.1)
    else:
        inputs[feat] = col.number_input(FEATURE_LABELS[feat], lo, hi, DEFAULTS[feat], step=1)

predict = st.button("Predict MACE risk", type="primary", use_container_width=True)

if predict:
    vals = [inputs[f] for f in FEATURES]
    # neut_pct is stored as a 0–1 proportion in training; input is a percentage, so divide by 100
    vals[FEATURES.index("neut_pct")] = vals[FEATURES.index("neut_pct")] / 100.0
    X = np.array([vals], dtype=float)
    X_scaled = scaler.transform(X)
    prob = float(model.predict_proba(X_scaled)[0, 1])

    st.divider()
    st.subheader("Prediction")

    if prob < 0.05:
        level, emoji = "Low risk", "🟢"
        note = "Below the cohort median; routine follow-up suggested."
    elif prob < 0.10:
        level, emoji = "Moderate risk", "🟡"
        note = "Around the cohort median; consider attention to kidney function and anemia."
    elif prob < 0.16:
        level, emoji = "Moderate-high risk", "🟠"
        note = "Above the cohort median; intensify cardiovascular risk-factor control."
    else:
        level, emoji = "High risk", "🔴"
        note = "In the upper quartile of the cohort; consider comprehensive cardio-kidney-metabolic intervention and specialist follow-up."

    c1, c2 = st.columns(2)
    with c1:
        st.metric("MACE risk probability", f"{prob:.1%}")
    with c2:
        st.metric("Risk stratum", f"{emoji} {level}")

    st.progress(min(prob / 0.30, 1.0), text="Relative risk bar (full ≈ 30%)")
    st.info(note)

    st.caption("Direction of each feature's contribution to risk (model coefficients):")
    coefs = {f: c for f, c in zip(FEATURES, model.coef_[0])}
    contrib = []
    for f in FEATURES:
        direction = "↑ risk" if coefs[f] > 0 else "↓ risk"
        contrib.append(f"- {FEATURE_LABELS[f]}: {direction}")
    st.write("\n".join(contrib))

    st.divider()
    st.caption(
        "⚠️ For research and educational purposes only; not a clinical diagnostic or treatment decision tool. "
        "The model was developed on a retrospective cohort and outputs an event probability, not a deterministic outcome."
    )
