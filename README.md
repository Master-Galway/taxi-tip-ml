# NYC Taxi Tip Prediction (Machine Learning)

This project develops machine learning models to predict whether a New York City taxi customer will leave a **generous tip (≥ 20%)**. The goal is to explore how predictive modeling can help taxi drivers optimize their revenue, while also considering the ethical implications of such a system.

This project demonstrates my practical experience with:

- Python (Pandas, NumPy, Scikit-Learn, XGBoost)
- End-to-end data cleaning and feature engineering
- One-hot encoding of high-cardinality categorical data (location IDs)
- Model training, cross-validation, and hyperparameter tuning
- Evaluation using precision, recall, F1, confusion matrix, and accuracy
- Ethical analysis of model deployment in real-world environments
- Professional project structuring, reproducibility, and documentation

---

## Repository Structure

```
nyc-taxi-tip-prediction/
│
├── data/
│   ├── sample_taxi.csv
│   ├── sample_predicted_means.csv
│   └── README.md
│
├── notebooks/
│   └── ML_Taxi_Tip_Prediction.ipynb
│
├── reports/
│   ├── ML_Taxi_Tip.pdf
│   └── Taxi_Data_Dictionary.pdf
│
├── src/
│   └── train_model.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Project Overview

Taxi drivers depend heavily on tips, which vary widely depending on the passenger, route, traffic conditions, and unobserved behavioral factors. This project investigates whether historical trip data can help predict whether a rider is likely to tip generously.

The target label is defined as:
generous = 1 if tip_percent >= 0.20
generous = 0 otherwise


The dataset includes timestamped trip information, location IDs, passenger count, fare components, and precomputed statistical features.

---

## Feature Engineering Highlights

The following features were engineered to improve predictive performance:

### **Datetime Features**
- Day of week (`day`)
- Time-of-day buckets:
  - `am_rush` (06:00–10:00)
  - `daytime` (10:00–16:00)
  - `pm_rush` (16:00–20:00)
  - `nighttime` (20:00–06:00)
- Month abbreviation (`month`)

### **Categorical Encoding**
Converted to strings and then one-hot encoded:
- `RatecodeID`
- `VendorID`
- `PULocationID`
- `DOLocationID`

This creates several hundred binary features, but the dimensionality remains manageable for tree-based models.

---

## Modeling Approach

Two tree-based classifiers were trained and tuned:

### **1. Random Forest**
- Robust to noise  
- Handles high-dimensional sparse data  
- Performs implicit feature selection  
- Less sensitive to hyperparameters

### **2. XGBoost**
- Boosted decision trees with gradient boosting  
- Strong performance on tabular data  
- Requires careful hyperparameter tuning  
- More computationally expensive  

Both were optimized using **GridSearchCV** with 4-fold cross-validation, tracking:

- Accuracy  
- Precision  
- Recall  
- F1 score (used for model selection)

---

## Results Summary

| Model            | Precision | Recall | F1    | Accuracy |
|-----------------|-----------|--------|-------|-----------|
| Random Forest (CV) | ~0.693 | ~0.786 | ~0.737 | ~0.725 |
| Random Forest (Test) | **~0.698** | **~0.790** | **~0.742** | **~0.731** |
| XGBoost (CV)     | ~0.693 | ~0.782 | ~0.734 | ~0.724 |
| XGBoost (Test)   | ~0.699 | ~0.789 | ~0.741 | ~0.730 |

**Key Insight:**  
Despite XGBoost typically outperforming Random Forest on many datasets, the results here are extremely similar. This is expected because:

- The dataset is moderately sized  
- Features are well structured  
- Both models handle sparse categorical encoding well  

The **Random Forest model slightly outperformed XGBoost** in this case.

---

## Confusion Matrix Interpretation (Random Forest)

Visually summarized in the notebook and PDF.

A typical output (example):

|               | Predicted 0 | Predicted 1 |
|---------------|-------------|-------------|
| **Actual 0**  | 1052        | 509         |
| **Actual 1**  | 313         | 1179        |

Interpretation:

- The model identifies **generous tippers** (class 1) reasonably well.
- False positives (509) indicate predicting a generous tip when one did not occur.
- False negatives (313) indicate missing generous tippers.
- Accuracy ≈ 73% shows meaningful predictive power for behavioral data.

---

## Ethical Considerations

The project includes discussion on:

### **False positives**
- Drivers may expect higher tips than they actually receive.

### **False negatives**
- Customers who *would* tip well may inadvertently be deprioritized.

### Ethical stance
Any real-world deployment should be carefully designed to avoid:
- unfair discrimination,
- overreliance on algorithmic predictions,
- misuse of behavioral inference.

This framework aligns with responsible AI principles.

---

## Data Source

Original trip data from the NYC Taxi & Limousine Commission (TLC):  
https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page

Only a **small sample** is included in this repository.

---

## Author

**Wei Gao**  
Data Scientist 
GitHub: https://github.com/Master-Galway
LinkedIn: https://www.linkedin.com/in/galway-gao/
