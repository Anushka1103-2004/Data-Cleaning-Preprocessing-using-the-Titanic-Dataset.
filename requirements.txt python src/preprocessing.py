## 📊 Output
A clean, processed dataset ready for machine‑learning models.

# 📄 src/preprocessing.py
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler


# Load dataset
df = pd.read_csv("../data/titanic.csv")
print("Initial Shape:", df.shape)


# Basic info
df.info()
print(df.isnull().sum())


# Handling missing values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Fare'].fillna(df['Fare'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)


# Encoding categorical variables
label = LabelEncoder()
df['Sex'] = label.fit_transform(df['Sex'])
df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)


# Outlier Removal using IQR
for column in ['Age', 'Fare']:
Q1 = df[column].quantile(0.25)
Q3 = df[column].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
df = df[(df[column] >= lower) & (df[column] <= upper)]


# Feature Scaling
scaler = StandardScaler()
df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])


# Save cleaned dataset
df.to_csv('../data/titanic_cleaned.csv', index=False)
print("Cleaning Completed. Final Shape:", df.shape)                                
