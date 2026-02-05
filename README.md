# 🫀 Heart Disease Prediction using Machine Learning

## Overview

This repository implements a **binary classification system** for predicting the **presence or absence of heart disease** using clinical and diagnostic patient data.  
The project follows a **production-oriented machine learning workflow**, including exploratory data analysis (EDA), feature engineering, model benchmarking, threshold optimization, and deployment-ready inference.

Multiple algorithms were evaluated, and **XGBoost** was selected as the final model due to its superior performance, probability calibration, and robustness on structured medical data.

---

## Problem Statement

Cardiovascular disease is one of the leading causes of mortality worldwide. Early detection through predictive modeling can assist clinicians in risk stratification and decision-making.

**Goal:**  
Predict whether a patient has heart disease (`Presence` / `Absence`) based on demographic, clinical, and diagnostic features.

---

## Dataset Description

### Target Variable
- **Heart Disease**
  - `0` → Absence
  - `1` → Presence

### Features
| Feature | Description |
|------|------------|
| Age | Patient age |
| Sex | Biological sex |
| Chest pain type | Chest pain classification |
| BP | Resting blood pressure |
| Cholesterol | Serum cholesterol |
| FBS over 120 | Fasting blood sugar > 120 mg/dl |
| EKG results | Resting ECG results |
| Max HR | Maximum heart rate achieved |
| Exercise angina | Exercise-induced angina |
| ST depression | ST depression induced by exercise |
| Slope of ST | Slope of peak exercise ST segment |
| Number of vessels fluro | Number of major vessels via fluoroscopy |
| Thallium | Thallium stress test result |

---

## Exploratory Data Analysis (EDA)

The following analyses were performed:

- Distribution analysis of numerical variables
- Correlation analysis to detect multicollinearity
- Outlier inspection using boxplots
- Cross-tabulation between categorical features and the target variable
- Class balance analysis

### Key Observations
- Strong associations between heart disease and:
  - Exercise-induced angina
  - Thallium stress test results
  - ST segment–related features
- Certain numerical features exhibited skewness and outliers, handled through model robustness rather than aggressive trimming.

---

## Models Evaluated

Three classification models were trained and evaluated using consistent preprocessing and metrics:

### 1. Logistic Regression
**Purpose:** Baseline interpretable model  
- Pros:
  - Simple and interpretable
  - Fast training
- Cons:
  - Limited capacity for non-linear relationships
  - Lower ROC-AUC and recall on positive cases

### 2. Random Forest
**Purpose:** Non-linear ensemble benchmark  
- Pros:
  - Captures feature interactions
  - Robust to outliers
- Cons:
  - Weaker probability calibration
  - Slightly lower ROC-AUC compared to boosting methods

### 3. XGBoost (Final Model) ✅
**Purpose:** High-performance gradient boosting classifier

---

## Why XGBoost Was Selected

XGBoost consistently outperformed other models across all critical evaluation metrics:

| Metric | Logistic Regression | Random Forest | **XGBoost** |
|------|--------------------|---------------|-------------|
| Accuracy | Moderate | High | **Highest** |
| ROC-AUC | Lower | High | **Highest (~0.94+)** |
| Recall (Positive Class) | Moderate | High | **Highest & Tunable** |
| Probability Calibration | Weak | Moderate | **Strong** |
| Feature Interaction Handling | Poor | Good | **Excellent** |

### Key Reasons
- Superior performance on structured medical data
- Well-calibrated probability outputs
- Effective handling of non-linear relationships
- Industry-standard model for tabular prediction problems

---

## Threshold Optimization

Rather than relying on a default 0.5 decision threshold, multiple thresholds were evaluated:

- **0.3** → High recall (screening-focused)
- **0.4** → Balanced precision-recall (selected)
- **0.5** → Conservative predictions

Final threshold selection was guided by:
- Clinical relevance
- False-negative risk minimization
- Precision–recall trade-offs

---

## Model Evaluation (Validation Set)

- **Accuracy:** ~89%
- **ROC-AUC:** ~0.94
- **Balanced Precision and Recall**
- Stable performance across evaluated thresholds

These results indicate strong generalization and reliable predictive behavior.

---

## Test Set Inference

Since the test dataset does not contain target labels:

- Predictions were generated using the trained pipeline
- Probability scores were preserved for interpretability
- Class labels were assigned using the optimized threshold

### Outputs Generated
- `Heart_Disease_Probability`
- `Heart_Disease_Prediction` (0 / 1)

---

## Model Persistence

The final XGBoost pipeline (including preprocessing steps) was serialized using `joblib` to ensure reproducibility and deployment readiness.

```python
joblib.dump(xgb_pipeline, "xgboost_heart_disease_model.pkl")
