import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


#importing dataset
data = pd.read_csv('C:/Users/Admin/Desktop/ML/Assignment_2/mobile_sales.csv')
print(data)

#checking for missing data
print()
print(data.isnull().sum())

#handling missing values if any

numerical_col= data.select_dtypes(include=[np.number]).columns

imputer = SimpleImputer(strategy='median')
data[numerical_col] = imputer.fit_transform(data[numerical_col])

#verify if missing values are handled
print() 
print(data.isnull().sum())
print() 
print(data)

#select numerical columns (replace with actual column names)
numerical_features = ['Sales_2023','Sales_2024','Sales_Maharashtra']

x= data[numerical_features]
print()
print(x)

#standardize the data

x= StandardScaler().fit_transform(x)

pca= PCA(n_components=3) #choose the number of components you want to keep

principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data = principalComponents , columns =['principal component 1','principal component 2', 'principal component 3'])

print()
print(principalDf)



# Based on the plot, choose the number of components
# For example, if you want components that explain 95% of the variance:
explained_variance_threshold = 1
n_components = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= explained_variance_threshold) + 1

print(f"\nNumber of components explaining {explained_variance_threshold * 100}% of the variance: {n_components}")

# Reduce data using the chosen number of components
pca = PCA(n_components=n_components)
principalComponents = pca.fit_transform(x)

principalDf = pd.DataFrame(data=principalComponents,
                           columns=[f'principal component {i+1}' for i in range(n_components)])

print("\nPrincipal Components DataFrame:\n", principalDf)

