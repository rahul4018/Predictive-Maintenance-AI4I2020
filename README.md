# Predictive Maintenance — AI4I 2020 Dataset

> Predicting machine failures before they happen using the AI4I 2020 benchmark dataset. End-to-end: data exploration, feature engineering, model training, and evaluation.

---

## Overview

Unplanned machine downtime costs manufacturing businesses billions annually. Predictive maintenance uses sensor data to flag machines likely to fail — before they do — enabling scheduled maintenance instead of emergency repairs.

This project applies classical machine learning to the [AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset), a realistic synthetic dataset designed to mirror real industrial conditions.

---

## What's covered

- **Exploratory Data Analysis** — distributions, correlations, class imbalance analysis, failure type breakdown
- **Data preprocessing** — handling imbalance (SMOTE), feature scaling, encoding
- **Model training** — Logistic Regression, Random Forest, XGBoost, compared systematically
- **Evaluation** — accuracy, precision, recall, F1, ROC-AUC; confusion matrices; feature importance
- **Insights** — which sensor readings matter most, and what failure modes they predict

---

## Dataset

| Property | Value |
|---|---|
| Source | UCI ML Repository |
| Rows | 10,000 synthetic data points |
| Features | 14 (air temperature, process temperature, rotational speed, torque, tool wear, etc.) |
| Target | Machine failure (binary) + failure type (5 types) |
| Class balance | ~97% no-failure / ~3% failure (imbalanced) |

Failure types covered: Tool Wear Failure (TWF), Heat Dissipation Failure (HDF), Power Failure (PWF), Overstrain Failure (OSF), Random Failure (RNF).

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.10+ | Language |
| Jupyter Notebook | Exploration & documentation |
| pandas / NumPy | Data manipulation |
| scikit-learn | ML models, preprocessing, metrics |
| imbalanced-learn | SMOTE for class balancing |
| matplotlib / seaborn | Visualisation |
| XGBoost | Gradient boosting model |

---

## Getting Started

```bash
git clone https://github.com/rahul4018/Predictive-Maintenance-AI4I2020.git
cd Predictive-Maintenance-AI4I2020

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

jupyter notebook
```

Open `notebooks/predictive_maintenance.ipynb` to follow the full analysis.

---

## Project Structure

```
Predictive-Maintenance-AI4I2020/
├── data/
│   └── ai4i2020.csv          # Dataset
├── notebooks/
│   └── predictive_maintenance.ipynb   # Main notebook
├── models/
│   └── best_model.pkl        # Saved trained model
├── requirements.txt
└── README.md
```

---

## Results (summary)

| Model | Accuracy | F1 (failure class) | ROC-AUC |
|---|---|---|---|
| Logistic Regression | ~96% | ~0.61 | ~0.87 |
| Random Forest | ~98% | ~0.82 | ~0.97 |
| XGBoost | ~98% | ~0.84 | ~0.97 |

Random Forest and XGBoost significantly outperform Logistic Regression on the minority (failure) class — which is what actually matters in a maintenance context.

---

## Key Takeaways

1. **Class imbalance is the main challenge** — a model that predicts "no failure" for everything gets 97% accuracy and is completely useless
2. **Tool wear and torque are the strongest failure predictors** across all models
3. **SMOTE improves recall on the failure class** without tanking precision significantly
4. **XGBoost edges out Random Forest** on F1 after tuning — but both are production-viable

---

## References

- [AI4I 2020 Dataset — UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset)
- Matzka, S. (2020). Explainable Artificial Intelligence for Predictive Maintenance Applications.

---

## License

MIT
