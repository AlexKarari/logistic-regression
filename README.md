# Logistic Regression

## Binary Classification — Telco Customer Churn

Built a Logistic Regression classifier using gradient descent, applied to a real-world telecom customer churn dataset (7,043 customers, 20 features).

---

## Problem Statement

Given a telecom customer's account details, service subscriptions, and billing history, predict whether they will **churn** (cancel their service).

This is a **binary classification** problem with natural class imbalance (~26% churn rate).

## Dataset

**Source:** [Kaggle — Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (IBM sample dataset)

| Feature Type | Examples | Encoding |
|---|---|---|
| **Demographic** | gender, SeniorCitizen, Partner, Dependents | Binary (0/1) |
| **Services** | InternetService, OnlineSecurity, TechSupport, StreamingTV... | One-hot encoded |
| **Account** | Contract, PaymentMethod, PaperlessBilling | One-hot encoded |
| **Numeric** | tenure (months), MonthlyCharges ($), TotalCharges ($) | StandardScaler |
| **Target** | Churn (Yes/No) | Binary (0/1) |

**Data cleaning challenges handled:**
- `TotalCharges` contains empty strings (not NaN) for new customers → coerced to 0.0
- Mixed encoding strategies needed for binary vs multi-category columns
- `drop_first=True` for one-hot encoding to avoid multicollinearity

**Final feature count after encoding: 30**

## Technical Approach

### Algorithm: Logistic Regression

| Component | Formula |
|-----------|---------|
| Hypothesis | ŷ = σ(θᵀx) |
| Sigmoid | σ(z) = 1 / (1 + e⁻ᶻ) |
| Cost (Cross-Entropy) | J(θ) = -(1/m)Σ[y·log(ŷ) + (1-y)·log(1-ŷ)] |
| L2 Regularization | + (λ/2m)Σθ² |
| Gradient | ∂J/∂θ = (1/m)Xᵀ(ŷ - y) + (λ/m)θ |

### Implementation Highlights

- **Numerically stable sigmoid** — conditional computation prevents overflow
- **L2 regularization** — optional Ridge penalty to control weight magnitudes
- **Probability calibration** — `predict_proba()` returns meaningful probabilities
- **Vectorized NumPy** — no explicit Python loops over samples
- **Real data pipeline** — cleaning, encoding, scaling, train/test split

## Performance

| Metric | Train | Test |
|--------|-------|------|
| Accuracy | ~0.80 | ~0.80 |
| Precision | ~0.66 | ~0.65 |
| Recall | ~0.54 | ~0.53 |
| F1 Score | ~0.60 | ~0.58 |

*Validated against sklearn LogisticRegression — exact metric match.*

### Top Churn Predictors (learned weights)

| Feature | Direction | Interpretation |
|---------|-----------|----------------|
| Contract (Two year) | ↓ churn | Long contracts = loyalty |
| tenure | ↓ churn | Longer tenure = less likely to leave |
| Contract (One year) | ↓ churn | Any contract reduces churn |
| Fiber optic | ↑ churn | Price or quality dissatisfaction? |
| Electronic check | ↑ churn | Less "committed" payment method |
| MonthlyCharges | ↑ churn | Higher bills = more churn |

## Key Findings

1. **Contract type dominates** — month-to-month customers churn at ~42% vs ~3% for two-year contracts
2. **Fiber optic paradox** — premium service but highest churn rate, suggesting price/quality mismatch
3. **Tenure is protective** — each additional month significantly reduces churn probability
4. **Electronic check = warning sign** — highest churn rate among payment methods (~45%)
5. **Cross-entropy converges smoothly** — validates the convexity advantage over MSE for classification

## ML Concepts Practiced

- **Sigmoid function** and numerical stability
- **Cross-entropy loss** and why MSE fails for classification
- **Categorical encoding** — label encoding vs one-hot and when to use each
- **Real data cleaning** — hidden missing values, type coercion
- **Class imbalance awareness** — accuracy alone is misleading at 26% minority class
- **Feature scaling** for gradient descent convergence
- **L2 regularization** effect on weight magnitudes

## Limitations

- **Linear decision boundary** — Logistic Regression assumes features combine linearly
- **No feature engineering** — interaction terms (e.g., tenure × contract) could improve performance
- **Threshold = 0.5** — may not be optimal for imbalanced data 
- **No cross-validation** — single train/test split (future improvement)

## Quick Start

```bash
# 1. Navigate to project
cd logistic-regression

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download dataset from Kaggle
#    → https://www.kaggle.com/datasets/blastchar/telco-customer-churn
#    → Place Telco-Customer-Churn.csv in data/
#    (Or run: python generate_mock_data.py for testing)

# 4. Train model
python train_model.py

# 5. Explore the notebook
jupyter notebook notebooks/logistic_regression.ipynb
```
