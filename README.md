Car Price Prediction with Machine Learning

Project Overview
This project predicts used car prices using machine learning. It includes data cleaning, exploratory analysis, feature engineering, and model comparison to accurately estimate vehicle values based on features like age, mileage, fuel type, and transmission.

Key Features
Data cleaning pipeline removing bikes and outliers

Comprehensive EDA with visualizations

Feature engineering (vehicle age calculation)

Comparison of 5 regression models

Hyperparameter tuning with GridSearchCV

Feature importance analysis

Prediction function for new inputs

Automated PDF report generation

Dataset
The dataset contains used car listings with:

Target: Selling_Price

Features: Present_Price, Driven_kms, Fuel_Type, Selling_type, Transmission, Owner, Year

Cleaned size: 290 records × 8 features

Requirements
Python 3.6+

pandas, numpy

scikit-learn

matplotlib, seaborn

joblib (for model saving)

reportlab (for PDF reports)

Install with:

bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib reportlab
Usage
Run the analysis:

bash
python carprice.py
The script will:

Clean and preprocess the data

Generate visualizations

Train and compare models

Save the best model as car_price_model.pkl

Create a PDF report

Make new predictions:

python
from predict import predict_car_price

price = predict_car_price(
    Present_Price=5.59,
    Driven_kms=27000,
    Fuel_Type='Petrol',
    Selling_type='Dealer',
    Transmission='Manual',
    Owner=0,
    Vehicle_Age=2
)
Results
Best performing model: Random Forest

RMSE: 1.72

R² Score: 0.91

Key findings:

Present price and vehicle age are most important features

Diesel and automatic transmission cars retain higher value

First-owner cars command 5-8% premium

