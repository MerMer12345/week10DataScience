import pandas as pd
from sklearn.impute import SimpleImputer

# Load your dataset
df = pd.read_csv("Cancer Dataset.csv")

# Drop 'Unnamed: 32' if it's irrelevant
if 'Unnamed: 32' in df.columns:
    df = df.drop(columns=['Unnamed: 32'])

# Identify numerical columns
numerical_cols = df.select_dtypes(include=['number']).columns.tolist()

# Exclude columns with all NaN values
numerical_cols = [col for col in numerical_cols if df[col].notna().sum() > 0]

# Initialize the SimpleImputer
imputer = SimpleImputer(strategy='mean')

# Apply the imputer to numerical columns
# Ensure the resulting DataFrame matches the structure of the original DataFrame
df[numerical_cols] = pd.DataFrame(
    imputer.fit_transform(df[numerical_cols]),
    columns=numerical_cols,
    index=df.index
)

# If you want to save the cleaned dataset to a new file
df.to_csv("cleaned_dataset.csv", index=False)

# Print a preview of the cleaned dataset
print(df.head())