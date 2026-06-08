Group 13 - AI3013 Machine Learning Course Project
Telecom Customer Churn Prediction Using Machine Learning Models

1. Project Overview
This project predicts telecom customer churn using classical machine learning models
implemented from scratch. The problem is a binary classification task: predict whether
a customer will churn based on demographic, service, contract, and billing attributes.

2. Submission Contents
Final_Report/
  Group13_Telco_Churn_Final_Report.pdf
  Group13_Telco_Churn_Final_Report.docx

Presentation/
  Telco_Churn_Presentation_Final.pptx

Code/
  Group13_Telco_Churn_From_Scratch.py
  WA_Fn-UseC_-Telco-Customer-Churn.csv
  requirements.txt
  outputs/

Dataset_Link.txt
Submission_Checklist.txt

3. Implementation Notes
The code implements the following models from scratch:
- Logistic Regression
- Cost-Sensitive Logistic Regression
- Gaussian Naive Bayes
- Hybrid Naive Bayes
- Decision Tree

The implementation does not use scikit-learn, TensorFlow, Keras, or any prebuilt
machine learning model library. It uses only:
- NumPy
- Pandas
- Matplotlib

4. How to Run the Code
Recommended environment:
- Python 3.9 or newer
- numpy
- pandas
- matplotlib

Install dependencies if needed:
pip install -r Code/requirements.txt

Run from the submission root:
cd Code
python Group13_Telco_Churn_From_Scratch.py

The dataset file must remain in the same Code folder:
WA_Fn-UseC_-Telco-Customer-Churn.csv

5. Expected Outputs
The script loads and preprocesses the dataset, performs EDA, trains all five models,
runs stratified 5-fold cross-validation, evaluates multiple metrics, performs threshold
analysis, analyzes generalization gap, and generates figures/tables in Code/outputs/.

6. Main Experimental Conclusion
Cost-Sensitive Logistic Regression achieves the best F1-score and is recommended when
the business objective is to identify more high-risk churn customers for retention.
Standard Logistic Regression achieves the highest accuracy and strong ROC-AUC.
Gaussian Naive Bayes achieves the highest recall but produces more false positives.

7. Reproducibility Notes
The random seed is fixed in the source code. Preprocessing avoids data leakage by fitting
standardization parameters inside each training fold only.
