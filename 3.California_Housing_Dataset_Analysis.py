import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the California housing dataset
california_housing = fetch_california_housing(as_frame=True)
california_df = pd.concat([california_housing.data, california_housing.target], axis=1)
california_df.rename(columns={'target': 'MedHouseVal'}, inplace=True)  # Rename the target column

#Title
# Display heading and subheading with user's name
st.title('California Housing Dataset Analysis')
st.subheader('Analyzing relationships with quantities - by Lokesh Dandumahanti')

st.subheader('Change the variables present in the sidebar to the left')


# Display the first few rows of the dataset
st.write(california_df.head())

# Sidebar for user input
st.sidebar.title('Model Parameters')
test_size = st.sidebar.slider('Test Size', 0.1, 0.5, 0.2, 0.1)
random_state = st.sidebar.slider('Random State', 0, 100, 42)

# Extract features (X) and target variable (y)
X = california_df.drop('MedHouseVal', axis=1)
y = california_df['MedHouseVal']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Build a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f'Mean Squared Error: {mse:.2f}')
st.write(f'R-squared: {r2:.2f}')

# Plotting predicted vs actual values
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred)
ax.set_xlabel('Actual Prices')
ax.set_ylabel('Predicted Prices')
ax.set_title('Actual vs Predicted Prices')
st.pyplot(fig)

# Residual Analysis
residuals = y_test - y_pred
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=residuals, color='blue', ax=ax)
ax.axhline(0, color='red', linestyle='--', linewidth=2)
ax.set_title('Residual Analysis')
ax.set_xlabel('Predicted Prices')
ax.set_ylabel('Residuals')
st.pyplot(fig)
