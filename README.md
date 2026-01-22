# Machine Learning Based Credit Card Fraud Detection

## Project Description
This repository contains a comprehensive analysis and implementation of machine learning models designed to identify fraudulent credit card transactions. The primary challenge addressed in this project is the detection of anomalies within a highly imbalanced dataset, where fraudulent activities represent a tiny fraction of legitimate transactions.

## Methodology

The project follows a structured data science pipeline:

1.  **Data Exploration & Visualization**: 
    - Analyzed the distribution of transaction amounts and time.
    - Investigated the class imbalance ratio.
    - Visualized correlations between features to identify key predictors.

2.  **Preprocessing & Sampling**:
    - Addressed class imbalance using **SMOTE (Synthetic Minority Over-sampling Technique)** and other sampling strategies to prevent model bias towards the majority class.
    - Scaled and normalized data for optimal model performance.

3.  **Model Development**:
    - Implemented multiple classification algorithms to benchmark performance:
        - **Logistic Regression**: A baseline linear model.
        - **Random Forest Classifier**: An ensemble learning method for robust predictions.
        - **XGBoost**: A high-performance gradient boosting framework.

4.  **Evaluation Metrics**:
    - Due to the skewed nature of the data, standard accuracy is insufficient. This project focuses on:
        - **Precision & Recall**: To balance false positives and false negatives.
        - **F1-Score**: The harmonic mean of precision and recall.
        - **ROC-AUC**: Area under the Receiver Operating Characteristic curve.

## Repository Contents

| File | Description |
|------|-------------|
| `analysis.ipynb` | The main Jupyter Notebook containing the full workflow: EDA, preprocessing, training, and evaluation code. |
| `best_fraud_model.pkl` | Serialized Python object of the highest-performing model, ready for inference. |
| `creditcard.csv` | The source dataset containing anonymized transaction records. |
| `flagged_transactions.csv` | Output file listing transactions classified as fraudulent by the model. |

## Requirements

To replicate the analysis, the following Python libraries are required:

- `pandas` (Data manipulation)
- `numpy` (Numerical computations)
- `matplotlib` & `seaborn` (Data visualization)
- `scikit-learn` (Machine learning models and metrics)
- `xgboost` (Gradient boosting implementation)

## Usage Instructions

1.  **Clone the repository** to your local machine.
2.  **Install dependencies**:
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn xgboost
    ```
3.  **Run the analysis**:
    Open `analysis.ipynb` in Jupyter Notebook or JupyterLab to execute the cells sequentially.
4.  **Inference**:
    Load the `best_fraud_model.pkl` to make predictions on new transaction data.

---
*Note: The dataset used contains anonymized features (V1-V28) resulting from a PCA transformation to protect user confidentiality.*
