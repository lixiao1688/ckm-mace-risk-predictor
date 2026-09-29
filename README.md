# CKM 综合征 MACE 风险预测 — Streamlit 应用

基于三中心 CKD 队列（千佛山推导 + 中心医院/省立外验）开发的**简约 6 特征机器学习模型**的在线预测工具。

## 运行

```bash
cd streamlit_app
pip install -r requirements.txt
streamlit run app.py
```

浏览器自动打开 `http://localhost:8501`。

## 模型

| 项目 | 值 |
|---|---|
| 模型 | 逻辑回归（LASSO 惩罚，C=0.01，class_weight=None） |
| 特征 | age、neutrophil%、RBC、creatinine、eGFR、合并症数（Boruta 筛选） |
| 内部 AUC | 0.768（5 折交叉验证） |
| 外部验证 | 省立 0.752（5 特征）、中心医院 0.808（非化验 2 特征子集） |
| 推导队列 | 千佛山医院 3,225 例（MACE 389，12.1%） |

## 特征说明

| 特征 | 单位 | 说明 |
|---|---|---|
| age | 岁 | 年龄 |
| neut_pct | % | 中性粒细胞百分比（炎症） |
| rbc | ×10¹²/L | 红细胞计数（贫血） |
| crea | µmol/L | 血肌酐 |
| egfr | mL/min/1.73 m² | CKD-EPI 估算肾小球滤过率 |
| n_comorb | 0–10 | 合并症计数 |

## 风险分层阈值（基于训练集分布）

- 低风险：< 5%
- 中风险：5%–10%
- 中高风险：10%–16%
- 高风险：> 16%

## 免责声明

本工具仅供科研与教学参考，不构成临床诊断或治疗决策依据。
