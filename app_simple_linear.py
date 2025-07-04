# app.py

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def load_and_train_model_simple_linear():
    # Load dataset
    df = pd.read_csv('Salary_Data.csv')
    X = df[['YearsExperience']]
    y = df['Salary']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=0)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model, X_train, X_test, y_train, y_test, df

# Set page config
st.set_page_config(page_title="💰 Salary Prediction App", layout="centered")

# Title and description
st.title("💰 Salary Prediction App")
st.markdown("Enter years of experience to predict salary or view the regression model.")

# Load model and data
model, X_train, X_test, y_train, y_test, df = load_and_train_model_simple_linear()

# Sidebar input
st.sidebar.header("Predict Salary")
years_exp = st.sidebar.number_input("Years of Experience", min_value=0.0, max_value=50.0, value=5.0, step=0.1)

# Make prediction
prediction = model.predict([[years_exp]])

# Display prediction
st.subheader("Predicted Salary")
st.write(f"Based on {years_exp} years of experience, the predicted salary is: **${prediction[0]:,.2f}**")

# Show regression equation
st.subheader("Regression Equation")
coef = model.coef_[0]
intercept = model.intercept_
st.latex(f"\\text{{Salary}} = {coef:,.2f} \\times \\text{{YearsExperience}} + {intercept:,.2f}")

# Plotting the regression line
st.subheader("Salary vs Experience (Regression Line)")
fig, ax = plt.subplots()
ax.scatter(X_train, y_train, color='red', label='Actual Data')
ax.plot(X_train, model.predict(X_train), color='blue', label='Regression Line')
ax.set_title('Salary vs Experience')
ax.set_xlabel('Years of Experience')
ax.set_ylabel('Salary')
ax.legend()
st.pyplot(fig)

# Option to show raw data
if st.checkbox("Show Raw Data"):
    st.subheader("Raw Dataset")
    st.write(df)