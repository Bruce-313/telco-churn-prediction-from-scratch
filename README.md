# Telecom Customer Churn Prediction Using Machine Learning Models From Scratch

## 1. Project Overview

This project predicts telecom customer churn using classical machine learning models implemented from scratch. The task is a binary classification problem: predicting whether a customer will churn based on demographic, service, contract, and billing information.

## 2. Repository Contents

This repository contains the following files:

* `Telco_Churn_From_Scratch.py`
  Main Python source code for data preprocessing, model implementation, model training, cross-validation, evaluation, and visualization.

* `Final_Report.pdf`
  Final project report, including project background, dataset analysis, methodology, experimental results, and conclusions.

* `Dataset_Link.txt`
  Dataset source and access information.

* `README.md`
  Project documentation.

## 3. Models Implemented From Scratch

The project implements the following machine learning models from scratch:

* Logistic Regression
* Cost-Sensitive Logistic Regression
* Gaussian Naive Bayes
* Hybrid Naive Bayes
* Decision Tree

The implementation does not use scikit-learn, TensorFlow, Keras, or other prebuilt machine learning model libraries.

## 4. Tools and Libraries

This project uses:

* Python
* NumPy
* Pandas
* Matplotlib

## 5. How to Run the Code

Recommended environment:

* Python 3.9 or newer
* numpy
* pandas
* matplotlib

Install the required libraries:

pip install numpy pandas matplotlib

Download the dataset according to the information in `Dataset_Link.txt`.

Place the dataset CSV file in the same folder as:

`Telco_Churn_From_Scratch.py`

Then run the Python file:

python Telco_Churn_From_Scratch.py

## 6. Expected Outputs

The script loads and preprocesses the dataset, performs exploratory data analysis, trains five machine learning models, runs stratified 5-fold cross-validation, evaluates different performance metrics, performs threshold analysis, analyzes generalization performance, and generates result figures or tables.

## 7. Main Experimental Conclusion

Cost-Sensitive Logistic Regression performs well for identifying high-risk churn customers and is useful when the business goal is customer retention.

Standard Logistic Regression achieves strong overall accuracy and stability.

Gaussian Naive Bayes achieves high recall, but it may also produce more false positives.

## 8. Reproducibility Notes

The random seed is fixed in the source code. The preprocessing process avoids data leakage by fitting standardization parameters inside each training fold only.
