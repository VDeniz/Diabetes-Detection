Diabetes Detection Project

Overview
The Diabetes Detection Project is an advanced deep learning tool designed to predict diabetes risk using medical health indicators, tailored for clinical decision-making. By leveraging convolutional neural networks (CNN) and Vision Transformers (ViT), the project identifies high-risk patients, provides interpretable risk scores, and visualizes results through interactive dashboards. It has been tested for scalability, handling over 250,000 records on a system with specifications including Microsoft Windows 11, a 14-core Intel i9-12900H processor, and 16 GB of RAM, while remaining adaptable to systems with lower specifications.

Problem Statement
Early detection of diabetes is critical for timely intervention, yet traditional tools often struggle with large medical datasets, lack advanced deep learning capabilities, or fail to provide interpretable risk scores for physicians. This project addresses these challenges by integrating deep learning models, statistical analysis, and interactive visualizations to deliver accurate diabetes risk predictions, enabling targeted interventions and improved patient outcomes.

Features and Execution Workflow
3.1 Data Loading and Preprocessing
Medical data is ingested using Pandas, with preprocessing steps including removal of missing values, normalization of features (e.g., BMI, blood pressure), and validation of illogical entries (e.g., negative BMI). A derived feature, BMI_to_Age, is created to enhance model performance, and SMOTE is applied to balance class distributions.
3.2 Statistical Analysis
Logistic regression is used to compute feature weights for a diabetes risk score, while Pearson correlation analyzes the sensitivity of features to the risk score, identifying key predictors like BMI and age.
3.3 Deep Learning Models
Two deep learning models are implemented using PyTorch:

Convolutional Neural Network (CNN): Extracts local patterns from medical features for accurate classification.
Vision Transformer (ViT): Models complex relationships between features for robust predictions.

3.4 Model Evaluation
Models are evaluated using Accuracy and F1-Score metrics, ensuring reliable performance on imbalanced medical data. A subset of 50,000 records is used for training, and 10,000 for testing.
3.5 Visualization and Reporting
Interactive dashboards are generated using Plotly, displaying distributions of BMI, risk scores, and prediction probabilities, alongside a scatter plot of risk scores vs. probabilities. A PDF report summarizes findings and provides actionable insights for physicians.

Data Requirements
The project requires a dataset in CSV format with the following columns:

Diabetes_binary: Binary target (0 for non-diabetic, 1 for diabetic).
BMI: Body Mass Index (float).
Age: Patient age (integer).
Other health indicators (e.g., blood pressure, cholesterol levels).

The dataset should be placed at C:/Diabetes_Detection/diabetes_binary_health_indicators_BRFSS2015.csv and is expected to contain over 250,000 records to align with the project's scalability testing.

Diabetes Risk Score Metric
The Diabetes Risk Score is computed using weights from a logistic regression model. Let ( X ) represent the feature matrix (e.g., BMI, age), and ( w ) the normalized weights from logistic regression. The risk score is calculated as:
[\text{Risk Score} = X \cdot w]
This score quantifies the likelihood of diabetes for each patient, with higher scores indicating greater risk. Sensitivity analysis using Pearson correlation further identifies features with the strongest influence on the risk score, aiding clinical interpretation.

Dependencies
To run the project, ensure the following prerequisites are met:

Python: Version 3.8 or 3.9.
Pandas: Version 1.5.3 (for tabular data manipulation).
NumPy: Version 1.24.3 (for numerical computations).
Scikit-learn: Version 1.5.0 (for preprocessing and evaluation metrics).
PyTorch: Version 2.0.1 (for deep learning models).
Plotly: Version 5.22.0 (for interactive visualizations).
Matplotlib: Version 3.7.1 (for initial visualizations).
Seaborn: Version 0.12.2 (for enhanced statistical graphics).
SciPy: Version 1.13.1 (for Pearson correlation).
FPDF: Version 1.7.2 (for PDF report generation).
Imbalanced-learn: Version 0.10.1 (for SMOTE data balancing).

Install the dependencies using:
pip install -r requirements.txt

How to Run the Code
6.1 Prerequisites

Ensure Python 3.8 or 3.9 is installed on your system.
Ensure an active internet connection for initial package installation.

6.2 Setup

Clone the repository:git clone <repository-url>

Navigate to the project directory:cd <project-directory>

Install dependencies:pip install -r requirements.txt

6.3 Data Preparation

Download the dataset file diabetes_binary_health_indicators_BRFSS2015.csv from the link provided in the "Output Files" section and place it in C:/Diabetes_Detection/. Ensure the dataset contains over 250,000 records with the required columns as specified in the "Data Requirements" section.

6.4 Execution

Run the script in Python:python Diabetes_Detection.py

Alternatively, execute in Jupyter Notebook by converting the script to a notebook or copying the code into cells.
The script processes the data, trains models, generates visualizations, and saves outputs in C:/Diabetes_Detection/. Execution may take a few minutes depending on system specifications.

6.5 View Outputs

Open the interactive dashboard (Diabetes_Prediction_Dashboard.html) in a web browser.
Review the detailed report at C:/Diabetes_Detection/Diabetes_Detection_Report.pdf.

6.6 Troubleshooting

Module Not Found: Ensure all dependencies are installed correctly using pip install -r requirements.txt.
File Not Found: Verify the dataset is placed at C:/Diabetes_Detection/diabetes_binary_health_indicators_BRFSS2015.csv and the output directory has write permissions.
Visualization Issues: Ensure your browser supports Plotly visualizations (e.g., use the latest version of Chrome or Firefox).

Output Files
The project generates the following output files for analysis and reporting:

Dataset: The dataset used for analysis is available on a public repository (e.g., UCI Machine Learning Repository). Download diabetes_binary_health_indicators_BRFSS2015.csv and place it in C:/Diabetes_Detection/.
Interactive Dashboard: C:/Diabetes_Detection/Diabetes_Prediction_Dashboard.html - An interactive dashboard displaying BMI, risk score, and probability distributions, with a scatter plot of risk scores vs. prediction probabilities.
PDF Report: C:/Diabetes_Detection/Diabetes_Detection_Report.pdf - A detailed report summarizing methodology, results, feature analysis, and clinical applications.
README: C:/Diabetes_Detection/README.md - This documentation file, providing an overview and instructions for the project.

Interacting with the Dashboard
The project provides an interactive dashboard (Diabetes_Prediction_Dashboard.html) for exploring results:

BMI Distribution: Visualizes the distribution of Body Mass Index values across the dataset.
Risk Score Distribution: Displays the distribution of computed diabetes risk scores.
Probability Distribution: Shows the distribution of model prediction probabilities.
Risk Score vs. Probability Scatter Plot: An interactive scatter plot where each point represents a patient, with hover data showing true labels. Colors range from dark purple (low probability) to red (high probability).

Users can hover over visualizations to view detailed data points and explore patterns in diabetes risk.

Contributing
Contributions to the Diabetes Detection Project are welcome. To contribute:

Fork the repository and create a new branch for your feature or bug fix.
Ensure your code adheres to the project's coding standards and includes appropriate documentation.
Submit a pull request with a detailed description of your changes.

For major changes, please open an issue first to discuss the proposed modifications.

License
This project is licensed under the MIT License. See the LICENSE file for details.

Note: Ensure the output directory (C:/Diabetes_Detection/) exists and has write permissions before running the script. For large datasets, ensure sufficient system resources (e.g., at least 8 GB RAM) to avoid performance issues. 
