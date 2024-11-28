# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, validation_curve, learning_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    mean_absolute_error, mean_squared_error, r2_score
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

# Split data into training and testing sets for validation curves, learning curves, and model comparison
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Hyperparameter Tuning using GridSearchCV for all models
# Define parameter grids for each model
param_grids = {
    "Random Forest": {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20],
        'min_samples_split': [2, 5, 10]
    },
    "SVM": {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf'],
        'gamma': ['scale', 'auto']
    },
    "Logistic Regression": {
        'C': [0.1, 1, 10],
        'solver': ['liblinear', 'saga']
    },
    "K-Nearest Neighbors": {
        'n_neighbors': [3, 5, 7, 10],
        'weights': ['uniform', 'distance']
    },
    "Naive Bayes": {
        'var_smoothing': [1e-9, 1e-8, 1e-7]
    }
}

# Initialize GridSearchCV for each model and perform hyperparameter tuning
best_models = {}

for model_name, param_grid in param_grids.items():
    print(f"Grid Search for {model_name}:")

    # Choose the model based on its name
    if model_name == "Random Forest":
        model = RandomForestClassifier(random_state=42)
    elif model_name == "SVM":
        model = SVC(random_state=42)
    elif model_name == "Logistic Regression":
        model = LogisticRegression(random_state=42)
    elif model_name == "K-Nearest Neighbors":
        model = KNeighborsClassifier()
    elif model_name == "Naive Bayes":
        model = GaussianNB()

    # Perform GridSearchCV
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Get and print the best parameters and store the best model
    best_models[model_name] = grid_search.best_estimator_
    print(f"Best Parameters for {model_name}: {grid_search.best_params_}")
    print("-" * 50)

# Step 4: Evaluate the Best Model from GridSearchCV using Cross-Validation
for model_name, model in best_models.items():
    cv_scores_best = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    print(f"Best Model ({model_name}) Cross-validation accuracy: {cv_scores_best.mean():.2f}")

# Step 5: Validation Curves for Hyperparameter Tuning
# Plot validation curve for each model and a relevant hyperparameter
param_ranges = {
    "Random Forest": {'param_range': [1, 2, 3, 4, 5], 'param_name': 'max_depth'},
    "SVM": {'param_range': [0.1, 1, 10], 'param_name': 'C'},
    "Logistic Regression": {'param_range': [0.1, 1, 10], 'param_name': 'C'},
    "K-Nearest Neighbors": {'param_range': [3, 5, 7, 10], 'param_name': 'n_neighbors'},
    "Naive Bayes": {'param_range': [1e-9, 1e-8, 1e-7], 'param_name': 'var_smoothing'}
}

for model_name, param_dict in param_ranges.items():
    print(f"Validation Curve for {model_name}:")

    model = best_models[model_name]

    # Generate validation curve
    train_scores, test_scores = validation_curve(
        model, X_train, y_train, param_name=param_dict['param_name'], param_range=param_dict['param_range'], cv=5
    )

    plt.plot(param_dict['param_range'], train_scores.mean(axis=1), label="Training score")
    plt.plot(param_dict['param_range'], test_scores.mean(axis=1), label="Test score")
    plt.xlabel(param_dict['param_name'])
    plt.ylabel("Score")
    plt.legend()
    plt.title(f"Validation Curve for {model_name}")
    plt.show()

# Step 6: Learning Curves for Each Model
for model_name, model in best_models.items():
    print(f"Learning Curve for {model_name}:")

    # Generate learning curve
    train_sizes, train_scores, test_scores = learning_curve(
        model, X_train, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 5)
    )

    plt.plot(train_sizes, train_scores.mean(axis=1), label="Training score")
    plt.plot(train_sizes, test_scores.mean(axis=1), label="Test score")
    plt.xlabel("Training Size")
    plt.ylabel("Score")
    plt.legend()
    plt.title(f"Learning Curve for {model_name}")
    plt.show()

# Step 7: Model Comparison
# Evaluate each model and print performance metrics
for model_name, model in best_models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.2f}")
    print(f"F1 Score: {f1:.2f}\n")
