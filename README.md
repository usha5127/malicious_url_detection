Malicious URL Detection System

This project implements a Machine Learning–based Malicious URL Detection System that classifies URLs as malicious or benign using feature engineering and a Random Forest classifier.

The project demonstrates an end-to-end ML pipeline including data preprocessing, feature extraction, model training, evaluation, and model persistence.

Project Overview

Malicious URLs are commonly used in phishing attacks, malware distribution, and online fraud.
This system analyzes structural characteristics of URLs and applies machine learning techniques to identify potentially harmful links.

Technologies Used

Python

Pandas, NumPy

Scikit-learn

Matplotlib, Seaborn

tldextract

Joblib

Machine Learning Workflow

Dataset loading (dataset.csv)

Data cleaning and preprocessing

URL-based feature extraction

Label encoding and feature encoding

Train-test split

Model training using Random Forest

Model evaluation (accuracy, classification report, confusion matrix)

Saving trained model and feature metadata

Project Structure

malicious-url-detection/
│── train_malicious_url_model.py
│── dataset.csv
│── cleaned_urls.csv
│── malicious_url_model.pkl
│── feature_columns.pkl
│── README.md

Note:
.pkl files are generated automatically after running the training script.

How to Run the Project
Step 1: Install Required Libraries
pip install pandas numpy scikit-learn matplotlib seaborn tldextract joblib

Step 2: Run the Training Script
python train_malicious_url_model.py

Step 3: Generated Output Files

After successful execution, the following files will be created:

cleaned_urls.csv

malicious_url_model.pkl

feature_columns.pkl

Model Output

Accuracy score

Classification report

Confusion matrix visualization

Trained model saved for future predictions

Dataset Description

The dataset contains the following columns:

url – URL string

label – Target class (malicious or benign)

A small dataset is used for academic and demonstration purposes.

Future Enhancements

Increase dataset size for improved accuracy

Add a Flask or Streamlit web interface

Deploy the model as a REST API

Use advanced NLP techniques for URL analysis

Author

Usha Rani Yanadhi
Bachelor of Technology – Computer Science and Engineering

Disclaimer

This project is intended for educational and academic use only and should not be considered a production-level security solution.