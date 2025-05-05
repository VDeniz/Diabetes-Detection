# Diabetes Detection Project

---

## Overview
The Diabetes Detection Project is an advanced deep learning tool designed to predict diabetes risk using medical health indicators, tailored for clinical decision-making. By leveraging convolutional neural networks (CNN) and Vision Transformers (ViT), the project identifies high-risk patients, provides interpretable risk scores, and visualizes results through interactive dashboards. It has been tested for scalability, handling over 250,000 records on a system with specifications including Microsoft Windows 11, a 14-core Intel i9-12900H processor, and 16 GB of RAM, while remaining adaptable to systems with lower specifications.

---

## Problem Statement
Early detection of diabetes is critical for timely intervention, yet traditional tools often struggle with large medical datasets, lack advanced deep learning capabilities, or fail to provide interpretable risk scores for physicians. This project addresses these challenges by integrating deep learning models, statistical analysis, and interactive visualizations to deliver accurate diabetes risk predictions, enabling targeted interventions and improved patient outcomes.

---

## Features and Execution Workflow

### 3.1 Data Loading and Preprocessing
Medical data is ingested using Pandas, with preprocessing steps including removal of missing values, normalization of features (e.g., BMI, blood pressure), and validation of illogical entries (e.g., negative BMI). A derived feature, `BMI_to_Age`, is created to enhance model performance, and SMOTE is applied to balance class distributions.

### 3.2 Statistical Analysis
Logistic regression is used to compute feature weights for a diabetes risk score, while Pearson correlation analyzes the sensitivity of features to the risk score, identifying key predictors like BMI and age.

### 3.3 Deep Learning Models
Two deep learning models are implemented using PyTorch:
- **Convolutional Neural Network (CNN)**: Extracts local patterns from medical features for accurate classification.
- **Vision Transformer (ViT)**: Models complex relationships between features for robust predictions.

### 3.4 Model Evaluation
Models are evaluated using Accuracy and F1-Score metrics, ensuring reliable performance on imbalanced medical data. A subset of 50,000 records is used for training, and 10,000 for testing.

### 3.5 Visualization and Reporting
Interactive dashboards are generated using Plotly, displaying distributions of BMI, risk scores, and prediction probabilities, alongside a scatter plot of risk scores vs. probabilities. A PDF report summarizes findings and provides actionable insights for physicians.

---

## Data Requirements
The project requires a dataset in CSV format with the following columns:
- `Diabetes_binary`: Binary target (0 for non-diabetic, 1 for diabetic).
- `BMI`: Body Mass Index (float).
- `Age`: Patient age (integer).
- Other health indicators (e.g., blood pressure, cholesterol levels).

The dataset should be placed at `C:/Diabetes_Detection/diabetes_binary_health_indicators_BRFSS2015.csv` and is expected to contain over 250,000 records to align with the project's scalability testing.

---

## Diabetes Risk Score Metric
The Diabetes Risk Score is computed using weights from a logistic regression model. Let \( X \) represent the feature matrix (e.g., BMI, age), and \( w \) the normalized weights from logistic regression. The risk score is calculated as:

\[
\text{Risk Score} = X \cdot w
\]

This score quantifies the likelihood of diabetes for each patient, with higher scores indicating greater risk. Sensitivity analysis using Pearson correlation further identifies features with the strongest influence on the risk score, aiding clinical interpretation.

---

## Dependencies
To run the project, ensure the following prerequisites are met:
- **Python**: Version 3.8 or 3.9.
- **Pandas**: Version 1.5.3 (for tabular data manipulation).
- **NumPy**: Version 1.24.3 (for numerical computations).
- **Scikit-learn**: Version 1.5.0 (for preprocessing and evaluation metrics).
- **PyTorch**: Version 2.0.1 (for deep learning models).
- **Plotly**: Version 5.22.0 (for interactive visualizations).
- **Matplotlib**: Version 3.7.1 (for initial visualizations).
- **Seaborn**: Version 0.12.2 (for enhanced statistical graphics).
- **SciPy**: Version 1.13.1 (for Pearson correlation).
- **FPDF**: Version 1.7.2 (for PDF report generation).
- **Imbalanced-learn**: Version 0.10.1 (for SMOTE data balancing).

Install the dependencies using:

```bash
pip install -r requirements.txt