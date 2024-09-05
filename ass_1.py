import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


data = {
    "Region": ["India", "America", "USA", "Italy", "USA", "Korean", "Brazil", "India", "USA", "South Africa"],
    "Age": [49, 32, 35, 43, 45, np.nan, np.nan, 53, 55, 42],
    "Income": [86400, 57600, 64800, 73200, np.nan, 69600, 62400, 94800, 99600, 80400],
    "Online Shopper": ["No", "Yes", "No", "No", "Yes", "Yes", "No", "Yes", "No", "Yes"]
}

df = pd.DataFrame(data)
print(df)

# Handling missing values 
df.fillna({'Age': df['Age'].mean(), 'Income': df['Income'].mean()}, inplace=True)

# Label Encoding 
label_encoder = LabelEncoder()
df['Online Shopper'] = label_encoder.fit_transform(df['Online Shopper'])

# One-Hot Encoding for the 'Region' column
one_hot_encoder = OneHotEncoder()
region_encoded = one_hot_encoder.fit_transform(df[['Region']]).toarray()
region_encoded_df = pd.DataFrame(region_encoded, columns=one_hot_encoder.get_feature_names_out(['Region']))

# Concatenating the encoded columns with the original DataFrame
df = pd.concat([df, region_encoded_df], axis=1).drop(['Region'], axis=1)

print("Encoded DataFrame:")
print(df)

# Define the feature set (X) and the target variable (y)
X = df.drop("Online Shopper", axis=1)
y = df["Online Shopper"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Optionally, print the first few rows of each dataset to inspect
print("\nTraining Features (X_train):")
print(X_train.head())
print("\nTesting Features (X_test):")
print(X_test.head())
print("\nTraining Labels (y_train):")
print(y_train.head())
print("\nTesing Labels (y_test):")
print(y_test.head())
