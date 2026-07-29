"""House Price Prediction — Linear Regression + Streamlit."""
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def create_dataset(rows: int = 1800) -> pd.DataFrame:
    """Generate reproducible demo housing data so no download is required."""
    rng = np.random.default_rng(42)
    area = rng.integers(500, 4501, rows)
    bedrooms = rng.integers(1, 6, rows)
    bathrooms = rng.integers(1, 5, rows)
    age = rng.integers(0, 41, rows)
    location = rng.choice(["City", "Suburb", "Rural"], rows, p=[.35, .50, .15])
    parking = rng.choice(["Yes", "No"], rows, p=[.70, .30])
    location_effect = pd.Series(location).map({"City": 110000, "Suburb": 55000, "Rural": 0}).to_numpy()
    parking_effect = np.where(parking == "Yes", 18000, 0)
    price = 60000 + area * 185 + bedrooms * 22000 + bathrooms * 17500 - age * 1800 + location_effect + parking_effect + rng.normal(0, 35000, rows)
    return pd.DataFrame({"area_sqft": area, "bedrooms": bedrooms, "bathrooms": bathrooms, "age_years": age,
                         "location": location, "parking": parking, "price": price.round(2)})


@st.cache_resource
def train_model():
    data = create_dataset()
    X, y = data.drop(columns="price"), data["price"]
    numerical = ["area_sqft", "bedrooms", "bathrooms", "age_years"]
    categorical = ["location", "parking"]
    preprocessor = ColumnTransformer([
        ("numbers", StandardScaler(), numerical),
        ("categories", OneHotEncoder(handle_unknown="ignore"), categorical),
    ])
    model = Pipeline([("preprocessor", preprocessor), ("regressor", LinearRegression())])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    return model, {"mae": mean_absolute_error(y_test, prediction), "r2": r2_score(y_test, prediction)}


st.set_page_config(page_title="House Price Predictor", page_icon="🏠")
model, metrics = train_model()
st.title("🏠 House Price Predictor")
st.caption("Estimate property prices with a Linear Regression model.")

with st.sidebar:
    st.header("Property details")
    area = st.number_input("Area (sq ft)", min_value=500, max_value=4500, value=1500, step=50)
    bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5], index=2)
    bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4], index=1)
    age = st.slider("Property age (years)", 0, 40, 8)
    location = st.selectbox("Location", ["City", "Suburb", "Rural"], index=1)
    parking = st.selectbox("Parking available", ["Yes", "No"])

property_data = pd.DataFrame([{"area_sqft": area, "bedrooms": bedrooms, "bathrooms": bathrooms,
                               "age_years": age, "location": location, "parking": parking}])
estimated_price = model.predict(property_data)[0]
st.metric("Estimated house price", f"${estimated_price:,.0f}")

st.subheader("Model evaluation")
col1, col2 = st.columns(2)
col1.metric("Mean Absolute Error", f"${metrics['mae']:,.0f}")
col2.metric("R² score", f"{metrics['r2']:.3f}")
st.info("This app uses a synthetic dataset for demonstration. Replace `create_dataset()` with a real, responsibly sourced housing dataset when extending the project.")
