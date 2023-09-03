# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset from a CSV file
data = pd.read_csv("E:\\New folder\\python\\Iris Flower - Iris.csv")

print("Dataset:\n", data.head())
print(data.tail())
print("\nShape of data\n",data.shape)

# Extract features (X) and target labels (y)
X = data.drop('Species', axis=1)
y = data['Species']

print("Features: \n",X.head())
print("Target: \n",y.head())

# Split the dataset into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a decision tree classifier
clf = DecisionTreeClassifier()

# Train the classifier on the training data
clf.fit(X_train, y_train)

# Make predictions on the test data
y_pred = clf.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=data['Species'].unique())

# Print the results
print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Classification Report:\n", report)
