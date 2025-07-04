# app.py

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Set page config
st.set_page_config(page_title="📊 Salary Prediction App", layout="centered")

# Title and description
st.title("📊 Salary Prediction Using Different Regression Models")
st.markdown("Select a regression model and predict salary for a given position level.")

# Load dataset
@st.cache_data
def load_data():
    dataset = pd.read_csv('Position_Salaries.csv')
    X = dataset.iloc[:, 1:-1].values
    y = dataset.iloc[:, -1].values
    return X, y

X, y = load_data()

# Sidebar - Model Selection
st.sidebar.header("Choose Regression Model")
model_choice = st.sidebar.selectbox(
    "Select a model",
    (
        "Linear Regression",
        "Polynomial Regression (degree=4)",
        "Decision Tree Regression",
        "Random Forest Regression",
        "SVR (Support Vector Regression)"
    )
)

# Sidebar input
position_level = st.sidebar.number_input("Enter Position Level (e.g., 6.5)", min_value=1.0, max_value=10.0, value=6.5, step=0.1)

# Placeholder for model and predictions
model = None
y_pred = None
X_grid = None
y_grid = None

# Train model based on selection
if model_choice == "Linear Regression":
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict([[position_level]])
    X_grid = np.arange(min(X), max(X), 0.1).reshape(-1, 1)
    y_grid = model.predict(X_grid)

elif model_choice == "Polynomial Regression (degree=4)":
    poly_reg = PolynomialFeatures(degree=4)
    X_poly = poly_reg.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(poly_reg.transform([[position_level]]))
    X_grid = np.arange(min(X), max(X), 0.1).reshape(-1, 1)
    y_grid = model.predict(poly_reg.transform(X_grid))

elif model_choice == "Decision Tree Regression":
    model = DecisionTreeRegressor(random_state=0)
    model.fit(X, y)
    y_pred = model.predict([[position_level]])
    X_grid = np.arange(min(X), max(X), 0.01).reshape(-1, 1)
    y_grid = model.predict(X_grid)

elif model_choice == "Random Forest Regression":
    model = RandomForestRegressor(n_estimators=10, random_state=0)
    model.fit(X, y)
    y_pred = model.predict([[position_level]])
    X_grid = np.arange(min(X), max(X), 0.01).reshape(-1, 1)
    y_grid = model.predict(X_grid)

elif model_choice == "SVR (Support Vector Regression)":
    sc_X = StandardScaler()
    sc_y = StandardScaler()
    X_scaled = sc_X.fit_transform(X)
    y_scaled = sc_y.fit_transform(y.reshape(-1, 1))
    model = SVR(kernel='rbf')
    model.fit(X_scaled, y_scaled.flatten())
    scaled_input = sc_X.transform([[position_level]])
    y_pred_scaled = model.predict(scaled_input)
    y_pred = sc_y.inverse_transform([y_pred_scaled])
    X_grid = np.arange(min(X), max(X), 0.1).reshape(-1, 1)
    X_grid_scaled = sc_X.transform(X_grid)
    y_grid_scaled = model.predict(X_grid_scaled)
    y_grid = sc_y.inverse_transform(y_grid_scaled.reshape(-1, 1))

# Show prediction
st.subheader("🔍 Prediction Result")
if model_choice == "SVR (Support Vector Regression)":
    st.write(f"For position level **{position_level}**, predicted salary is: **${float(y_pred[0]):,.2f}**")
else:
    st.write(f"For position level **{position_level}**, predicted salary is: **${y_pred[0]:,.2f}**")

# Plotting
fig, ax = plt.subplots()
ax.scatter(X, y, color='red', label='Actual Data')
ax.plot(X_grid, y_grid, color='blue', label=f'{model_choice} Fit')
ax.set_title(f"Truth or Bluff - {model_choice}")
ax.set_xlabel('Position Level')
ax.set_ylabel('Salary ($)')
ax.legend()
st.pyplot(fig)

# Optional: Show raw data
if st.checkbox("Show Raw Data"):
    st.subheader("Raw Dataset")
    st.dataframe(pd.DataFrame({'Level': X.flatten(), 'Salary': y}))