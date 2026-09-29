# -*- coding: utf-8 -*-
"""
CKM syndrome incident MACE risk prediction — parsimonious 6-feature model.
Streamlit app. Run:  streamlit run app.py
"""
import os
import pickle

import numpy as np
import streamlit as st

st.set_page_config(page_title="CKM-MACE 风险预测", page_icon="🫀", layout="centered")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")

FEATURE_LABELS = {
    "age": "年龄 (岁)",
    "neut_pct": "中性粒细胞百分比 (%)",
    "rbc": "红细胞计数 (×10¹²/L)",
    "crea": "肌酐 (µmol/L)",
    "egfr": "eGFR (mL/min/1.73 m²)",
    "n_comorb": "合并症数 (0–10)",
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

# 训练集预测概率分位数（用于风险分层）
TRAIN_QUANTILES = {"p10": 0.034, "p25": 0.055, "p50": 0.095, "p75": 0.159, "p90": 0.244}

st.title("🫀 CKM 综合征 MACE 风险预测")
st.markdown(
    "基于三中心 CKD 队列开发的**简约 6 特征机器学习模型**，预测新发主要不良心血管事件"
    "（MACE：心衰 / 心肌梗死 / 卒中）的 1 年风险。"
)

with st.sidebar:
    st.header("模型信息")
    st.markdown(
        """
        - **模型**：逻辑回归（LASSO 惩罚，C=0.01）
        - **特征**：6 个 Boruta 筛选的稳定预测因子
        - **内部 AUC**：0.768（5 折交叉验证）
        - **外部验证**：0.752（省立）/ 0.808（中心医院，非化验子集）
        - **校准 Brier**：0.094（全模型）
        - **推导队列**：千佛山医院 3,225 例（MACE 389，12.1%）
        """
    )
    st.caption("心血管-肾脏-代谢（CKM）综合征人群专用")

st.subheader("输入患者特征")
col1, col2 = st.columns(2)
inputs = {}
for i, feat in enumerate(FEATURES):
    col = col1 if i % 2 == 0 else col2
    lo, hi = MINMAX[feat]
    if isinstance(lo, float):
        inputs[feat] = col.number_input(FEATURE_LABELS[feat], lo, hi, DEFAULTS[feat], step=0.1)
    else:
        inputs[feat] = col.number_input(FEATURE_LABELS[feat], lo, hi, DEFAULTS[feat], step=1)

predict = st.button("预测 MACE 风险", type="primary", use_container_width=True)

if predict:
    vals = [inputs[f] for f in FEATURES]
    # neut_pct 训练数据为 0–1 比例，输入为百分比，需除以 100
    vals[FEATURES.index("neut_pct")] = vals[FEATURES.index("neut_pct")] / 100.0
    X = np.array([vals], dtype=float)
    X_scaled = scaler.transform(X)
    prob = float(model.predict_proba(X_scaled)[0, 1])

    st.divider()
    st.subheader("预测结果")

    # 风险分层
    if prob < 0.05:
        level, color = "低风险", "🟢"
        note = "风险低于队列中位水平，建议常规随访。"
    elif prob < 0.10:
        level, color = "中风险", "🟡"
        note = "风险处于队列中等水平，建议关注肾功能与贫血管理。"
    elif prob < 0.16:
        level, color = "中高风险", "🟠"
        note = "风险高于队列中位，建议加强心血管危险因素控制。"
    else:
        level, color = "高风险", "🔴"
        note = "风险位于队列上四分位，建议强化心肾代谢综合干预并专科随访。"

    c1, c2 = st.columns(2)
    with c1:
        st.metric("MACE 风险概率", f"{prob:.1%}")
    with c2:
        st.metric("风险分层", f"{color} {level}")

    st.progress(min(prob / 0.30, 1.0), text="相对风险条（满格≈30%）")
    st.info(note)

    # 特征贡献方向
    st.caption("各特征对风险的贡献方向（模型系数）：")
    coefs = {f: c for f, c in zip(FEATURES, model.coef_[0])}
    contrib = []
    for f in FEATURES:
        direction = "↑ 风险" if coefs[f] > 0 else "↓ 风险"
        contrib.append(f"- {FEATURE_LABELS[f]}：{direction}")
    st.write("\n".join(contrib))

    st.divider()
    st.caption(
        "⚠️ 本工具仅供科研与教学参考，不构成临床诊断或治疗决策依据。"
        "模型基于回顾性队列开发，预测为事件发生概率，非确定性结论。"
    )
