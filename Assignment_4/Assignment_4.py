# Importing all the libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import PolynomialFeatures

# Importing the dataset
dataset = pd.read_csv('spotify_2023_data.csv')
dataset.head()
print(dataset)
# Separate features and target (exclude the 'Language' column)
X = dataset[['Region', 'Artist', 'Song Length (min)']]  # Features: Region, Artist, Song Length
Y = dataset['Language'].apply(lambda x: 1 if x == 'English' else 0).values  # Labels: 1 for English, 0 for Hindi

# One-Hot Encoding for 'Region' and 'Artist'
column_transformer = ColumnTransformer(
    transformers=[
        ('onehot_region', OneHotEncoder(), ['Region']),  # Apply OneHotEncoder to the 'Region' column
        ('onehot_artist', OneHotEncoder(), ['Artist'])   # Apply OneHotEncoder to the 'Artist' column
    ], 
    remainder='passthrough'  # Leave 'Song Length' column unchanged
)

# Apply One-Hot Encoding to categorical variables
X_encoded = column_transformer.fit_transform(X)

# Add interaction features (Song Length interaction with Region and Artist)
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X_encoded)

# Data split for training and testing (75/25)
X_train, X_test, Y_train, Y_test = train_test_split(X_poly, Y, test_size=0.25, random_state=0)

# Scaling using StandardScaler for normal distribution (set with_mean=False for sparse matrices)
sc = StandardScaler(with_mean=False)
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Building the model using RBF kernel
classifier_rbf = SVC(kernel='rbf', random_state=0)
classifier_rbf.fit(X_train, Y_train)

# Predicting the test set results
Y_pred_rbf = classifier_rbf.predict(X_test)

# Printing the confusion matrix
cm_rbf = confusion_matrix(Y_test, Y_pred_rbf)
print(cm_rbf)

# Classification Report
class_report_rbf = classification_report(Y_test, Y_pred_rbf)
print(class_report_rbf)

# Accuracy Score
accuracy_rbf = accuracy_score(Y_test, Y_pred_rbf)
print("Accuracy Score: {:.2f}%".format(accuracy_rbf * 100))
