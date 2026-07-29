# House Price Prediction using Linear Regression

This project estimates house prices from property details through a **Linear Regression** model and an interactive Streamlit web app.

## Objective

Predict a property's estimated price using its area, number of bedrooms and bathrooms, age, location, and parking availability.

## Algorithm

**Linear Regression** is a supervised machine-learning algorithm for predicting continuous numeric values. Here, it learns the relationship between property features and house price.

## Features

- Area in square feet
- Number of bedrooms and bathrooms
- Property age
- Location category
- Parking availability

## Workflow

1. Generate or load housing data.
2. Scale numerical features and one-hot encode categorical features.
3. Split data into training and test datasets.
4. Train a Linear Regression model.
5. Evaluate predictions using Mean Absolute Error (MAE) and R² score.
6. Use Streamlit to predict a price from user-entered property details.

## Installation and run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open the Local URL shown in the terminal, normally `http://localhost:8501`.

## Project structure

```text
house-price-predictor/
├── app.py
├── requirements.txt
└── README.md
```

## Resume description

Developed a house-price prediction web application using Linear Regression, Scikit-learn preprocessing pipelines, and Streamlit. Applied feature scaling and categorical encoding, then evaluated regression performance using MAE and R² score to provide real-time property-price estimates.

## Note

The included dataset is generated programmatically for demonstration. Replace it with a real housing dataset to extend the project.
