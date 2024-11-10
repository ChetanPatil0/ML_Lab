# Implementation of Naive Bayes for Fake Job Posting

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Step 1: Load Dataset
df = pd.read_csv('C:/Users/Admin/Desktop/ML/Assignment_5/Gender_Detection.csv', encoding='ISO-8859-1')
print(df)

# Map gender to binary values
df['Gender'] = df['Gender'].map({'F': 0, 'M': 1})
print(df)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['Name'], df['Gender'], test_size=0.2, random_state=42)

# Convert the data into a bag-of-words model using CountVectorizer
vectorizer = CountVectorizer()
X_train_transformed = vectorizer.fit_transform(X_train)
X_test_transformed = vectorizer.transform(X_test)

# Step 3: Train the Naive Bayes Model
model = MultinomialNB()
model.fit(X_train_transformed, y_train)

# Step 4: Evaluate the Model
y_pred = model.predict(X_test_transformed)

# Print the accuracy
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Print the classification report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Print the confusion matrix
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
