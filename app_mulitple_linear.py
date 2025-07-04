# app.py

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Set page config
st.set_page_config(page_title="📈 Startup Profit Predictor", layout="centered")

# Title and description
st.title("📈 Startup Profit Prediction App")
st.markdown("Enter startup details to predict profit.")

# Load dataset
@st.cache_data
def load_data():
    dataset = pd.read_csv('50_Startups.csv')
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    return dataset, X, y

df, X, y = load_data()

# Preprocess data
@st.cache_resource
def preprocess_data(X, y):
    ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
    X_encoded = np.array(ct.fit_transform(X))

    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=0)

    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    return regressor, ct, X_test, y_test

model, ct, X_test, y_test = preprocess_data(X, y)

# Make prediction function
def predict_profit(model, ct, rnd, admin, marketing, state):
    # Prepare input as DataFrame to ensure correct column names
    input_data = pd.DataFrame([[
        rnd, admin, marketing, state
    ]], columns=['R&D Spend', 'Administration', 'Marketing Spend', 'State'])

    # Transform the 'State' column using ColumnTransformer
    encoded_state = ct.transform(input_data)[0]

    # Convert to 2D array for model prediction
    encoded_state = encoded_state.reshape(1, -1)

    # Predict profit
    predicted_profit = model.predict(encoded_state)
    return predicted_profit[0]


# Sidebar inputs
st.sidebar.header("Input Features")
rnd_spend = st.sidebar.number_input("R&D Spend", min_value=0.0, value=160000.0)
admin_spend = st.sidebar.number_input("Administration Spend", min_value=0.0, value=130000.0)
marketing_spend = st.sidebar.number_input("Marketing Spend", min_value=0.0, value=300000.0)
state = st.sidebar.selectbox("State", ("New York", "California", "Florida"))

# Predict button
if st.sidebar.button("Predict Profit"):
    predicted_profit = predict_profit(model, ct, rnd_spend, admin_spend, marketing_spend, state)
    st.success(f"Predicted Profit: **${predicted_profit:,.2f}**")

# Show regression equation
st.subheader("Regression Equation")
coef = model.coef_
intercept = model.intercept_

equation = "Profit = "
for i in range(len(coef)):
    if i < 3:
        states = ["New York", "California", "Florida"]
        equation += f"{coef[i]:.2f} × {states[i]} + "
    elif i == 3:
        equation += f"{coef[i]:.2f} × R&D + "
    elif i == 4:
        equation += f"{coef[i]:.2f} × Admin + "
    elif i == 5:
        equation += f"{coef[i]:.2f} × Marketing + "
st.latex(equation + f"{intercept:.2f}")

# Show actual vs predicted
if st.checkbox("Show Actual vs Predicted Values"):
    y_pred = model.predict(X_test)
    comparison = np.concatenate((y_pred.reshape(-1, 1), y_test.reshape(-1, 1)), axis=1)
    comparison_df = pd.DataFrame(comparison, columns=["Predicted", "Actual"])
    st.subheader("Actual vs Predicted Profits")
    st.dataframe(comparison_df.style.format("{:.2f}"))

# Show raw data
if st.checkbox("Show Raw Data"):
    st.subheader("Raw Dataset")
    st.dataframe(df)