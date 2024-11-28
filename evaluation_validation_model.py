
# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, validation_curve, learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

# Example dataset: Let's load the dataset (use your own dataset here)
# This is just an example, replace with the actual dataset loading
# For example, we use the Iris dataset here as a placeholder

from sklearn.datasets import load_iris

# Load the Iris dataset (for demonstration purposes)
data = load_iris()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Step 1: Train-Test Split
# Split the dataset into training and test sets (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Cross-Validation
# Perform k-fold cross-validation (typically 5 or 10 folds)
model = RandomForestClassifier(n_estimators=100, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=5)  # 5-fold cross-validation
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean cross-validation score: {cv_scores.mean():.2f}")

# Step 3: Performance Metrics
# Fit the model on the training data
model.fit(X_train, y_train)

# Predict using the trained model
y_pred = model.predict(X_test)

# Classification Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
conf_matrix = confusion_matrix(y_test, y_pred)

# Display the results
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"Confusion Matrix:\n{conf_matrix}")

# Step 4: Regression Metrics (if applicable, using RandomForestRegressor for demonstration)
# We will create an artificial regression problem here using random data.
# In your case, replace this with your regression target variable.

# For demonstration, let's generate some random regression data.
from sklearn.datasets import make_regression

X_reg, y_reg = make_regression(n_samples=100, n_features=5, noise=0.1, random_state=42)

# Split into train and test sets for regression
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)

# Create and fit a Random Forest Regressor
model_reg = RandomForestRegressor(n_estimators=100, random_state=42)
model_reg.fit(X_train_reg, y_train_reg)

# Predict using the regression model
y_pred_reg = model_reg.predict(X_test_reg)

# Regression Metrics
mae = mean_absolute_error(y_test_reg, y_pred_reg)
mse = mean_squared_error(y_test_reg, y_pred_reg)
r2 = r2_score(y_test_reg, y_pred_reg)

# Display regression results
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Step 5: Hyperparameter Tuning using GridSearchCV
# Perform grid search for hyperparameter tuning for the Random Forest model

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Get the best parameters from GridSearchCV
print(f"Best Parameters from GridSearch: {grid_search.best_params_}")
best_model = grid_search.best_estimator_

# Step 6: Evaluate the Best Model from GridSearchCV
y_pred_best = best_model.predict(X_test)

# Print the accuracy of the best model
print(f"Best Model Accuracy: {accuracy_score(y_test, y_pred_best):.2f}")

# Step 7: Validation Curves
# Plot validation curve for hyperparameter tuning (e.g., max_depth for Random Forest)

param_range = [1, 2, 3, 4, 5]
train_scores, test_scores = validation_curve(
    RandomForestClassifier(), X_train, y_train, param_name="max_depth", param_range=param_range, cv=5
)

plt.plot(param_range, train_scores.mean(axis=1), label="Training score")
plt.plot(param_range, test_scores.mean(axis=1), label="Test score")
plt.xlabel("Max Depth")
plt.ylabel("Score")
plt.legend()
plt.title("Validation Curve for Random Forest (Max Depth)")
plt.show()

# Step 8: Learning Curves
# Plot learning curve for Random Forest model

train_sizes, train_scores, test_scores = learning_curve(
    RandomForestClassifier(), X_train, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 5)
)

plt.plot(train_sizes, train_scores.mean(axis=1), label="Training score")
plt.plot(train_sizes, test_scores.mean(axis=1), label="Test score")
plt.xlabel("Training Size")
plt.ylabel("Score")
plt.legend()
plt.title("Learning Curve for Random Forest")
plt.show()

# Step 9: Model Comparison
# Compare the performance of different models

models = [
    ("Random Forest", RandomForestClassifier(n_estimators=100)),
    ("SVM", SVC(kernel='linear', random_state=42)),
    ("Logistic Regression", LogisticRegression(random_state=42)),
    ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=3)),
    ("Naive Bayes", GaussianNB())
]

# Evaluate each model and print performance metrics
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model: {name}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.2f}\n")

