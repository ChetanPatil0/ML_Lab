# a)importing libraries

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

# b)importing libraries

data= pd.read_csv('C:/Users/Admin/Desktop/ML/Assignment_1/Online_shopper.csv')

print(data)

# c)identifying and handling missing data

#identifying
print (data.isnull())
print(data.isnull().sum())

#handling (FILL WITH THE MODE OF THE COLUMN)
data['Age']=data['Age'].fillna(data['Age'].mode())
print(data)

#handling (FILL WITH THE MEAN OF THE COLUMN)
data['Income']=data['Income'].fillna(data['Income'].mean())
print(data)

# Verify if missing data is handled
print(data.isnull().sum())


#identify categorial columns

categorical_columns = data.select_dtypes(include=[np.number]).columns
print(categorical_columns)

#handling missing data by replacing with the mode (for categorical columns)

#categorical_columns = data.select_dtypes(include=[np.number]).columns

imputer = SimpleImputer (strategy='most_frequent')
data[categorical_columns]=imputer.fit_transform(data[categorical_columns])

#verify if missing data is handled
print(data.isnull().sum())

#Apply OneHotEnchoder to Categorical columns

encoder = OneHotEncoder(sparse_output=False,drop='first')

encoded_data = encoder.fit_transform(data[categorical_columns])

# Getting the names of encoded columns

encoded_columns=encoder.get_feature_names_out(categorical_columns)

#Drop original categorical columns and concatenate encoded columns

data =data.drop(categorical_columns,axis=1)
data=pd.concat([data,pd.DataFrame (encoded_data,columns=encoded_columns)],axis=1)

#Assuming 'Online shopping ' is the target variable, adjust if different

x=data.drop('Online Shopper',axis=1)
y=data['Online Shopper']

#e) split the data into training and testing sets(e.g., 80% train,20% test)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

print("Traning set Shape : ",x_train.shape,y_train.shape)
print("Testing set Shape : ",x_test.shape, y_test.shape)



















