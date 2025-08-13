# loan-prediction-model-2.0
Loan Prediction ML – End-to-end machine learning pipeline for predicting loan approval status using applicant demographics, income, credit history, and property details. Includes data preprocessing, EDA, multiple ML algorithms, model evaluation, and batch predictions.
#  Loan Prediction – Machine Learning Project

This repository contains a **comprehensive machine learning pipeline** designed to predict **loan approval status** based on applicant demographics, income, credit history, and property details.  

It leverages **multiple classification algorithms** to evaluate and compare performance, ensuring the most accurate and reliable model is selected for production.  
This is an **end-to-end solution** — from raw data ingestion to final model evaluation.

---

##  Project Objective

Loan approvals are critical decisions in the banking and fintech sectors. Manual processing:
- Is **time-consuming**
- Can be **biased**
- Is **hard to scale**

The objective is to:
- Automate loan approval prediction
- Reduce processing time
- Improve consistency and fairness in decision-making

---

## Data Set Overview

| Feature            | Description                                      |
| ------------------ | ------------------------------------------------ |
| Gender             | Applicant's gender                               |
| Married            | Marital status                                   |
| Dependents         | Number of dependents                             |
| Education          | Education level                                  |
| Self\_Employed     | Employment type                                  |
| ApplicantIncome    | Monthly income of applicant                      |
| CoapplicantIncome  | Monthly income of co-applicant                   |
| LoanAmount         | Loan amount requested                            |
| Loan\_Amount\_Term | Loan term in days                                |
| Credit\_History    | Credit history (1: good, 0: bad)                 |
| Property\_Area     | Urban, Semiurban, or Rural                       |
| Loan\_Status       | Target variable: Approved (Y) / Not Approved (N) |


##  Key Features

- **Data Preprocessing**
  - Handling missing values
  - Encoding categorical variables
  - Scaling numerical features
- **Exploratory Data Analysis (EDA)**
  - Insights into loan applicant profiles
  - Correlation heatmaps
  - Loan approval patterns
- **Multiple ML Algorithms**
  - Logistic Regression
  - Decision Tree Classifier
  - Random Forest Classifier
  - Gradient Boosting Classifier
  - K-Nearest Neighbors (KNN)
  - Support Vector Classifier (SVC)
  - Gaussian Naive Bayes
- **Evaluation Metrics**
  - Accuracy
  - Precision, Recall, F1 Score
  - Confusion Matrix
- **Batch Predictions**
  - Generates predictions for unseen test data

---

##  Tech Stack

| Layer           | Technology |
|-----------------|------------|
| Language        | Python     |
| Data Handling   | Pandas, NumPy |
| Visualization   | Matplotlib, Seaborn |
| ML Framework    | scikit-learn |
| Environment     | Jupyter Notebook |

---

## 📂 Repository Structure


---

## ⚙️ How to Run

1. **Clone the repository**
bash
git clone https://github.com/your-username/loan-prediction-ml.git
cd loan-prediction-ml

## Install Dependencies
pip install pandas numpy scikit-learn matplotlib seaborn jupyter

##Model Workflow
#Data Ingestion

Load training and test datasets

Exploratory Data Analysis

Identify correlations and patterns

Data Preprocessing

Missing value imputation

One-hot encoding

Model Training

Train multiple classification models:

Logistic Regression

Decision Tree

Random Forest

Gradient Boosting

KNN

SVC

Gaussian Naive Bayes

Model Evaluation

Accuracy score

Confusion matrix

Precision, recall, F1 score

Prediction

Apply best model to test_Y3wMUE5_7gLdaTN.csv

## Results Summary
The notebooks compare seven ML models and track their performance.
Final evaluation is done in accuracy_run.ipynb with metrics such as accuracy and confusion matrix.

Example (sample values):

Random Forest: ~80–85% accuracy

Gradient Boosting: ~78–82% accuracy

Logistic Regression: ~77–80% accuracy

(Random Forest often emerges as the top performer in classification problems with mixed categorical and numerical features.)

## Future Enhancements
Add XGBoost and LightGBM for boosting accuracy

Deploy as a Flask/Django API for real-time predictions

Build an interactive Streamlit dashboard for non-technical users

Integrate Explainable AI (XAI) for transparent loan decision-making
