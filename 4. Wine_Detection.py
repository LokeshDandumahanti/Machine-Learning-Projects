'''
A. BUILDING A MACHINE LEARNING MODEL


1. import modules
2. load data
3. divide data into x and y
4. divide data further into train and test with 80:20 ratio
5. find a suitable algorithm
6. fit the model and train it
7. use test data to predict.
8. compare it with actual test data
9. give the accuracy and classification report

B. ANALYSING DATA AND OBTAINING RELATIONAL PLOTS

1. get the data
2. store it in pandas dataframe form
3. preview of data
4. stats of data
5. drawing a correlation matrix
6. plot a graph individually to show relations between metrics

'''

#A. BUILDING A MACHINE LEARNING MODEL

#1. IMPORTING MODULES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

#2. LOAD DATA

wine = load_wine()

#3. DIVIDING DATA INTO X AND Y

X = wine.data
y = wine.target

#4. DIVIDE DATA INTO TRAIN AND TEST SETS

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state = 42)

#5. FIND SUITABLE ALGORITHM

algo = DecisionTreeClassifier()

#6. FIT THE MODEL AND TRAIN IT

algo.fit(X_train, y_train)

#7. USE DATA TO PREDICT IT

y_pred = algo.predict(X_test)

#8. COMPARE IT WITH ACTUAL TEST DATA

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

#9. GIVE THE ACCURACY

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

#10. GIVE THE OUTPUT

print("Accuracy:", accuracy)
print("\nClassification Report :\n", report)

#B. ANALYSING DATA AND OBTAINING RELATIONAL PLOTS

#2. STORE DATA IN PANDA FRAMES

wine_df = pd.DataFrame(data=wine.data, columns=wine.feature_names)
wine_df['target'] = wine.target

#3. PREVIEW OF DATA

print("\nSample of the wine dataset:")
print(wine_df.head())

#4. STATS OF DATA

print("\nSummary statistics of the Wine dataset:")
print(wine_df.describe())

#5. DRAWING OF CORRELATION MATRIX

correlation_matrix = wine_df.corr()
plt.figure(figsize = (10,8))
sns.heatmap(correlation_matrix, annot = True, 
            cmap = 'coolwarm', linewidths = 0.5)
plt.title("Correlation Matrix")
plt.show()

#6. PLOT OF INDIVIDUAL GRAPHS

selected_features = ['alcohol','flavanoids','color_intensity']
for feature_x in selected_features:
    for feature_y in selected_features:
        if feature_x != feature_y:
            sns.pairplot(wine_df[[feature_x, feature_y, 'target']], 
                         hue = 'target', palette = 'viridis', height = 5)
            plt.title(f"Pairplot of {feature_x} and {feature_y} with 
                      Target")
            plt.show()
            break


























