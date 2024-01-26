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

# Display the first few rows of the dataset
print(california_df.head())

# Summary statistics of the dataset
print(california_df.describe())

# Correlation matrix to analyze feature relationships
correlation_matrix = california_df.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()

# Pairplot to visualize relationships between variables
sns.pairplot(california_df[['MedHouseVal', 'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms']])
plt.show()

# Extract features (X) and target variable (y)
X = california_df.drop('MedHouseVal', axis=1)
y = california_df['MedHouseVal']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')

# Plotting predicted vs actual values
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()

# Residual Analysis
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_pred, y=residuals, color='blue')
plt.axhline(0, color='red', linestyle='--', linewidth=2)
plt.title('Residual Analysis')
plt.xlabel('Predicted Prices')
plt.ylabel('Residuals')
plt.show()
