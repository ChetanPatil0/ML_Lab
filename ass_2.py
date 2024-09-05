import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


os.chdir(r"C:\Users\MCA3\Desktop\42Chetan")


csv_file = pd.read_csv("chetan_1.csv")


print("*Print the dataframe**")
print(csv_file)
print("****************************************************")


print("Missing values in columns:")
print(csv_file.isna().sum())
print("****************************************************")
csv_file['Age'] = pd.to_numeric(csv_file['Age'], errors='coerce')
csv_file['Age'] = csv_file['Age'].fillna(csv_file['Age'].mean())

csv_file['Income'] = pd.to_numeric(csv_file['Income'], errors='coerce')
csv_file['Income'] = csv_file['Income'].fillna(csv_file['Income'].median())

csv_file['Region'] = csv_file['Region'].fillna(csv_file['Region'].mode()[0])

print("Cleaned dataframe:")
print(csv_file)
print("****************************************************")

# Split the dataset into features (X) and target (y)
X = csv_file.drop("Online Shopper", axis=1)
y = csv_file["Online Shopper"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
print("****************************************************")

# Feature Scaling 
scaler = StandardScaler()
X_train[['Age', 'Income']] = scaler.fit_transform(X_train[['Age', 'Income']])
X_test[['Age', 'Income']] = scaler.transform(X_test[['Age', 'Income']])

# Print the scaled features to verify
print("Scaled X_train:")
print(X_train.head())
print("****************************************************")
print("Scaled X_test:")
print(X_test.head())
print("****************************************************")
