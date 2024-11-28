# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Use inline plotting to prevent the HTTP 429 error in PyCharm
plt.ion()  # Turn on interactive mode

# Load dataset
df = pd.read_csv('Cancer Dataset.csv')

# Data Cleaning: Drop columns with all missing values
df = df.dropna(how='all', axis=1)

# Data Cleaning: Handle missing values
numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

# Example: Fill missing values in categorical columns with the mode
categorical_cols = df.select_dtypes(include=['object']).columns
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Exploratory Data Analysis (EDA)
print("Basic Info About the Dataset:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

# 1. Distribution of Numeric Features
numeric_df = df.select_dtypes(include=['number'])
if not numeric_df.empty:
    for col in numeric_df.columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[col], kde=True, bins=30, color='blue')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.show()
else:
    print("No numeric features available for plotting distributions.")

# 2. Count Plots for Categorical Variables
categorical_df = df.select_dtypes(include=['object'])
if not categorical_df.empty:
    for col in categorical_df.columns:
        plt.figure(figsize=(8, 4))
        # Set hue=None to avoid the warning
        sns.countplot(data=df, x=col, palette='viridis')
        plt.title(f'Count Plot of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.show()
else:
    print("No categorical features available for count plots.")

# 3. Correlation Heatmap
numeric_df = df.select_dtypes(include=['number'])
if not numeric_df.empty:
    plt.figure(figsize=(10, 6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
    plt.title("Correlation Heatmap")
    plt.show()
else:
    print("No numeric columns available for correlation heatmap.")

# 4. Pairplot for Feature Relationships
if not numeric_df.empty and numeric_df.shape[1] > 1:
    sns.pairplot(numeric_df)
    plt.suptitle("Pairplot of Numeric Features", y=1.02)
    plt.show()
else:
    print("Not enough numeric columns for pairplot.")

# 5. Box Plots for Outlier Detection
if not numeric_df.empty:
    for col in numeric_df.columns:
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=df, x=col, palette='Set2')
        plt.title(f'Box Plot of {col}')
        plt.xlabel(col)
        plt.show()
else:
    print("No numeric columns available for box plots.")

# 6. Handling Categorical Variables (Optional)
# Example: One-hot encode categorical columns if required
encoded_df = pd.get_dummies(df, drop_first=True)
print("\nDataset after One-Hot Encoding:")
print(encoded_df.head())

# Save the cleaned and encoded dataset to a new CSV file (optional)
# encoded_df.to_csv('cleaned_data.csv', index=False)

print("EDA Completed Successfully!")
