'''
1.import necessary libraries
2.load the dataset
3.show preview of data
3.splt the data into training and testing sets
3.check for missing values
3.visualise the data
4.create a algorithm model
5.Train the model
6.evaluate the model
7.display the results
'''

#1.import necessary libraries

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt

#2. Load the titanic dataset

titanic = sns.load_dataset('titanic')
print(titanic.head())
print(titanic.describe())
print(titanic.isnull().sum())

#3. visualise

# Survival based on passenger class
sns.countplot(x = 'survived', data = titanic)
plt.title('survival counts by passenger class')
plt.show()

# Survival based on gender
sns.countplot( x = 'sex', data = titanic)
plt.title('survival counts by gender')
plt.show()

#Survival based on age
sns.countplot(x='age', data = titanic)
plt.title('survival counts by age')
plt.show()

# Convert categorical variables to numerical
titanic = pd.get_dummies(titanic, columns=['sex', 'embarked', 'class', 'who', 'adult_male', 'alone'], drop_first=True)

# Drop unnecessary columns
titanic.drop(['deck', 'embark_town', 'alive'], axis=1, inplace=True)

# Clean the data: Fill missing values
titanic['age'].fillna(titanic['age'].median(), inplace=True)

# Fill missing values in the 'embarked_S' column
titanic['embarked_S'].fillna(titanic['embarked_S'].mode()[0], inplace=True)

# Drop 'survived' from X since it's the target variable
X = titanic.drop('survived', axis=1)

# Ensure 'y' is the 'survived' column
y = titanic['survived']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a Random Forest Classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# Display classification report
print('Classification Report:')
print(classification_report(y_test, y_pred))

