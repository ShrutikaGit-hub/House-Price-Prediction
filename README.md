# House Price Prediction using Linear Regression

This project predicts house prices based on property details using a Linear Regression machine learning model. It includes an interactive Streamlit web application where users can enter property information and receive an estimated house price.

## Objective

The objective of this project is to estimate the price of a house based on important property features such as area, bedrooms, bathrooms, location, property age, and parking availability.

## Algorithm Used

- Linear Regression

Linear Regression is a supervised machine learning algorithm used to predict continuous numerical values. In this project, it learns the relationship between house features and house price.

## Features Used

- Area in square feet
- Number of bedrooms
- Number of bathrooms
- Property age in years
- Location category
- Parking availability

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit

## Machine Learning Workflow

1. Generate or load the housing dataset.
2. Separate property features and target price.
3. Scale numerical features using `StandardScaler`.
4. Convert categorical features using one-hot encoding.
5. Split the dataset into training and testing data.
6. Train a Linear Regression model.
7. Evaluate the model using Mean Absolute Error (MAE) and R² score.
8. Predict property prices through the Streamlit web interface.

## Project Snap

<img width="1920" height="1200" alt="image" src="https://github.com/user-attachments/assets/8927d414-10bc-4c40-8e5e-d8e7b7c69754" />
