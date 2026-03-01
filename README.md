# Predictive Financial Modeling for Economic Inclusion

A machine learning project to predict credit card default risk using historical financial data.

---

## Overview

This project builds an end-to-end credit risk prediction pipeline trained on 30,000 real-world credit card client records. It applies data preprocessing, exploratory data analysis, feature engineering, multi-model comparison, and SHAP-based explainability to identify customers at high risk of default. The goal is to support transparent, data-driven financial decision-making.

---

## Dataset

**Name:** Default of Credit Card Clients Dataset
**Source:** UCI Machine Learning Repository
**Link:** https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
**File:** `default of credit card clients.xls`
**Records:** 30,000 customers
**Features:** 23 input variables + 1 target variable

| Feature Group | Variables |
|---|---|
| Credit Information | LIMIT_BAL |
| Demographics | SEX, EDUCATION, MARRIAGE, AGE |
| Repayment Status | PAY_0 to PAY_6 (monthly payment status) |
| Bill Statements | BILL_AMT1 to BILL_AMT6 |
| Payment History | PAY_AMT1 to PAY_AMT6 |
| Target | default.payment.next.month (1 = default, 0 = no default) |

Download the dataset from the link above and place the `.xls` file inside the `data/` folder before running the notebook.

---

## Repository Structure

```
├── notebooks/
│   └── credit_default_prediction.ipynb
├── data/
│   └── default of credit card clients.xls    (download separately — link above)
├── outputs/                                   (plots saved here when notebook runs)
├── requirements.txt
└── README.md
```

---

## Methodology

**1. Data Preprocessing**
Handled missing values, removed invalid category codes in EDUCATION and MARRIAGE columns, applied StandardScaler to numerical features, and performed a stratified 80/20 train-test split to preserve class balance.

**2. Exploratory Data Analysis**
Visualised class distribution to assess target imbalance (~22% default), plotted a correlation heatmap across all features, and analysed payment delay distributions across default and non-default customer groups.

**3. Feature Engineering**
Four new behavioral features were engineered from the raw data:
- `AVG_PAY_DELAY` — mean of PAY_0 to PAY_6, capturing sustained payment behavior
- `AVG_BILL_AMT` — mean of BILL_AMT1 to BILL_AMT6, smoothing monthly billing volatility
- `AVG_PAY_AMT` — mean of PAY_AMT1 to PAY_AMT6, reflecting repayment capacity
- `PAY_RATIO` — AVG_PAY_AMT divided by AVG_BILL_AMT, measuring how much of the bill is typically repaid

**4. Model Building**
Three classifiers were trained and compared: Logistic Regression (linear baseline), Random Forest Classifier (ensemble method), and Gradient Boosting Classifier (sequential boosting).

**5. Model Evaluation**
Each model was evaluated using Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Confusion Matrix analysis, with a focus on performance on the minority default class.

**6. SHAP Explainability**
SHAP (SHapley Additive exPlanations) was applied to the best-performing model to produce a global beeswarm summary plot and ranked feature importances, explaining which features drive default predictions and in which direction.

---

## Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 81.2% | 0.67 | 0.35 | 0.46 | 0.77 |
| Gradient Boosting | 82.1% | 0.68 | 0.42 | 0.52 | 0.79 |
| Random Forest | 82.4% | 0.70 | 0.45 | 0.55 | 0.80 |

Random Forest was selected as the final model. It achieved the best F1-Score on the default class and an ROC-AUC of 0.80 with stable cross-validated performance.

**Top SHAP features (Random Forest):**

| Rank | Feature | Insight |
|---|---|---|
| 1 | AVG_PAY_DELAY | Chronic payment delay is the strongest default signal |
| 2 | PAY_0 | Most recent payment status dominates individual predictions |
| 3 | LIMIT_BAL | Higher credit limits reduce default probability |
| 4 | AVG_PAY_AMT | Larger consistent payments indicate lower risk |
| 5 | PAY_2 | Payment delay from two months prior still carries weight |

The engineered feature AVG_PAY_DELAY ranked first above all 23 original variables. Demographic features (age, sex, education) ranked at the bottom, confirming the model's decisions are driven by financial behavior rather than personal characteristics.

---

## Tech Stack

- Python 3.8+
- Pandas, NumPy — data manipulation
- Matplotlib, Seaborn — visualisation
- Scikit-learn — model training and evaluation
- SHAP — model explainability
- Jupyter Notebook

---

## Getting Started

```bash
git clone https://github.com/goelavi04/Predictive-Financial-Modeling-for-Economic-Inclusion.git
cd Predictive-Financial-Modeling-for-Economic-Inclusion
pip install -r requirements.txt
jupyter notebook notebooks/credit_default_prediction.ipynb
```

Run all cells in order. Plots are saved automatically to the `outputs/` folder.

---

## License

MIT License
