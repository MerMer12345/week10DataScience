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

# Step 1: Cross-Validation (Instead of train-test split, using cross-validation on the full dataset)
model = RandomForestClassifier(n_estimators=100, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=5)  # 5-fold cross-validation
print(f"Cross-validation scores for Random Forest: {cv_scores}")
print(f"Mean cross-validation score: {cv_scores.mean():.2f}")

# Step 2: Performance Metrics with Cross-Validation
# Instead of fitting on train and predicting on test, we use cross_val_score to get performance metrics
models = [
    ("Random Forest", RandomForestClassifier(n_estimators=100)),
    ("SVM", SVC(kernel='linear', random_state=42)),
    ("Logistic Regression", LogisticRegression(random_state=42)),
    ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=3)),
    ("Naive Bayes", GaussianNB())
]

for name, model in models:
    # Perform 5-fold cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"Model: {name}")
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.2f}\n")

# Step 3: Hyperparameter Tuning using GridSearchCV for the Random Forest model
# Perform grid search for hyperparameter tuning for the Random Forest model

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)

# Get the best parameters from GridSearchCV
print(f"Best Parameters from GridSearch: {grid_search.best_params_}")
best_model = grid_search.best_estimator_

# Step 4: Evaluate the Best Model from GridSearchCV using Cross-Validation
cv_scores_best = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
print(f"Best Model (GridSearch) Cross-validation accuracy: {cv_scores_best.mean():.2f}")

# Step 5: Validation Curves
# Plot validation curve for hyperparameter tuning (e.g., max_depth for Random Forest)

param_range = [1, 2, 3, 4, 5]
train_scores, test_scores = validation_curve(
    RandomForestClassifier(), X, y, param_name="max_depth", param_range=param_range, cv=5
)

plt.plot(param_range, train_scores.mean(axis=1), label="Training score")
plt.plot(param_range, test_scores.mean(axis=1), label="Test score")
plt.xlabel("Max Depth")
plt.ylabel("Score")
plt.legend()
plt.title("Validation Curve for Random Forest (Max Depth)")
plt.show()

# Step 6: Learning Curves
# Plot learning curve for Random Forest model

train_sizes, train_scores, test_scores = learning_curve(
    RandomForestClassifier(), X, y, cv=5, train_sizes=np.linspace(0.1, 1.0, 5)
)

plt.plot(train_sizes, train_scores.mean(axis=1), label="Training score")
plt.plot(train_sizes, test_scores.mean(axis=1), label="Test score")
plt.xlabel("Training Size")
plt.ylabel("Score")
plt.legend()
plt.title("Learning Curve for Random Forest")
plt.show()
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

# Step 1: Cross-Validation (Instead of train-test split, using cross-validation on the full dataset)
model = RandomForestClassifier(n_estimators=100, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=5)  # 5-fold cross-validation
print(f"Cross-validation scores for Random Forest: {cv_scores}")
print(f"Mean cross-validation score: {cv_scores.mean():.2f}")

# Step 2: Performance Metrics with Cross-Validation
# Instead of fitting on train and predicting on test, we use cross_val_score to get performance metrics
models = [
    ("Random Forest", RandomForestClassifier(n_estimators=100)),
    ("SVM", SVC(kernel='linear', random_state=42)),
    ("Logistic Regression", LogisticRegression(random_state=42)),
    ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=3)),
    ("Naive Bayes", GaussianNB())
]

for name, model in models:
    # Perform 5-fold cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    print(f"Model: {name}")
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV accuracy: {cv_scores.mean():.2f}\n")

# Step 3: Hyperparameter Tuning using GridSearchCV for the Random Forest model
# Perform grid search for hyperparameter tuning for the Random Forest model

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Initialize GridSearchCV
grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=param_grid, cv=5, scoring='accuracy')
grid_search.fit(X, y)

# Get the best parameters from GridSearchCV
print(f"Best Parameters from GridSearch: {grid_search.best_params_}")
best_model = grid_search.best_estimator_

# Step 4: Evaluate the Best Model from GridSearchCV using Cross-Validation
cv_scores_best = cross_val_score(best_model, X, y, cv=5, scoring='accuracy')
print(f"Best Model (GridSearch) Cross-validation accuracy: {cv_scores_best.mean():.2f}")

# Split data into training and testing sets for validation curves, learning curves, and model comparison
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Validation Curves
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

# Step 6: Learning Curves
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

# Step 7: Model Comparison
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
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Model: {name}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1 Score: {f1:.2f}\n")
