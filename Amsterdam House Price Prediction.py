#!/usr/bin/env python
# coding: utf-8

# In[1]:


# AMSTERDAM HOUSE PRICE PREDICTION

# importing necessary libraries
import numpy as np   
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RepeatedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

df_ams=pd.read_csv('amsterdamMerged.csv')  # loading the dataset
df_ams.head()
df_ams.info()
df_ams.dtypes
df_ams.columns
df_ams.describe()

df_ams.rename(columns = {'AREA IN m²':'AREA IN SQ METRE'}, inplace = True)   #Renaming the column name 'AREA IN m²' with 'AREA IN SQ METRE'

df_ams.drop(columns=['TITLE',   # removing columns from the dataset that are not taken for further analysis and prediction
                     'File',
                     'URL',
                     'OFFERED SINCE',
                     'CONTACT DETAILS',
                     'TIMESTAMP',
                     'DESCRIPTION'],axis=1,inplace=True)

# MISSING VALUES
df_ams['VOLUME'] = df_ams['VOLUME'].replace('Volume is not available',np.nan)   # replacing all non available datas with NaN,
df_ams['INTERIOR'] = df_ams['INTERIOR'].replace('Interior is not available',np.nan)  #since this can help make it easier to work with and analyze the dataset, 
df_ams['AVAILABILITY'] = df_ams['AVAILABILITY'].replace('Not available to book',np.nan)  #and can help ensure that the data is accurate and complete.
df_ams['GARAGE'] = df_ams['GARAGE'].replace('Details of garage is not available',np.nan)
df_ams['UPKEEP STATUS'] = df_ams['UPKEEP STATUS'].replace('Upkeep is not available',np.nan)
df_ams['SPECIFICATION'] = df_ams['SPECIFICATION'].replace('Specifics are not available',np.nan)     
df_ams['LOCATION TYPE'] = df_ams['LOCATION TYPE'].replace('Location type is not available',np.nan)
df_ams['NUMBER OF FLOORS'] = df_ams['NUMBER OF FLOORS'].replace('Number of floors is not available',np.nan)
df_ams['DETAILS OF GARDEN'] = df_ams['DETAILS OF GARDEN'].replace('Details of garden is not available',np.nan)
df_ams['DETAILS OF STORAGE'] = df_ams['DETAILS OF STORAGE'].replace('Details of storage is not available',np.nan)
df_ams['NUMBER OF BEDROOMS'] = df_ams['NUMBER OF BEDROOMS'].replace('Number of bedrooms is not available',np.nan)
df_ams['DETAILS OF BALCONY'] = df_ams['DETAILS OF BALCONY'].replace('Details of balcony is not available',np.nan)
df_ams['NUMBER OF BATHROOMS'] = df_ams['NUMBER OF BATHROOMS'].replace('Number of bathrooms is not available',np.nan)
df_ams['DESCRIPTION OF STORAGE'] = df_ams['DESCRIPTION OF STORAGE'].replace('Details of description of the storage is not available',np.nan)

df_col_null = df_ams.columns[df_ams.isna().any()==True].tolist()  # creating a list of columns with null values to fill them with appropriate values and also to drop columns with less available data

df_ams.drop(columns=['AVAILABILITY','SPECIFICATION','LOCATION TYPE','DESCRIPTION OF STORAGE'],axis=1,inplace=True)  # dropping columns with less available data,i.e., columns with more than 70% of NaN values

df_ams['VOLUME'] = df_ams['VOLUME'].astype(float) #converting the datatype of 'VOLUME' into float for replacing the NaN values of 'VOLUME' with mean of the column values

df_ams['VOLUME'].fillna(df_ams['VOLUME'].mean(),inplace=True)  #replacing the NaN values in 'VOLUME' with mean value of the available data in the column
df_ams['GARAGE'].fillna(df_ams['GARAGE'].mode()[0],inplace=True)  #replacing the NaN values in 'GARAGE' with mode of the available data in the column
df_ams['INTERIOR'].fillna(df_ams['INTERIOR'].mode()[0],inplace=True)  #replacing the NaN values in 'INTERIOR' with mode of the available data in the column
df_ams['UPKEEP STATUS'].fillna(df_ams['UPKEEP STATUS'].mode()[0],inplace=True)  #replacing the NaN values in 'UPKEEP STATUS' with mode of the available data in the column
df_ams['NUMBER OF FLOORS'].fillna(df_ams['NUMBER OF FLOORS'].mode()[0],inplace=True)  #replacing the NaN values in 'NUMBER OF FLOORS' with mode of the available data in the column
df_ams['DETAILS OF GARDEN'].fillna(df_ams['DETAILS OF GARDEN'].mode()[0],inplace=True)  #replacing the NaN values in 'DETAILS OF GARDEN' with mode of the available data in the column
df_ams['DETAILS OF BALCONY'].fillna(df_ams['DETAILS OF BALCONY'].mode()[0],inplace=True)  #replacing the NaN values in 'DETAILS OF BALCONY' with mode of the available data in the column
df_ams['DETAILS OF STORAGE'].fillna(df_ams['DETAILS OF STORAGE'].mode()[0],inplace=True)  #replacing the NaN values in 'DETAILS OF STORAGE' with mode of the available data in the column
df_ams['NUMBER OF BEDROOMS'].fillna(df_ams['NUMBER OF BEDROOMS'].mode()[0],inplace=True)  #replacing the NaN values in 'NUMBER OF BEDROOMS' with mode of the available data in the column
df_ams['NUMBER OF BATHROOMS'].fillna(df_ams['NUMBER OF BATHROOMS'].mode()[0],inplace=True)  #replacing the NaN values in 'NUMBER OF BATHROOMS' with mode of the available data in the column 

# FEATURE ENGINEERING

# 1.DETAILS OF GARDEN (# details like availability of garden(present/Not present),area of garden,location of garden can be extracted from this column)
df_ams[["AVAILABILITY OF GARDEN",'AREA OF GARDEN']] = df_ams["DETAILS OF GARDEN"].str.split("(", expand = True) # splitting the column with '(' into 2 different columns 
df_ams['AVAILABILITY OF GARDEN']=df_ams['AVAILABILITY OF GARDEN'].replace(['Present '],'Present') #removing unnecessary characters from data
df_ams[['AREA OF GARDEN IN SQ METRE','g1']] = df_ams["AREA OF GARDEN"].str.split("m²", expand = True)  #splitting the column 'Area of Garden' with 'm²' into 2 different columns of area and location
df_ams.drop(columns=['AREA OF GARDEN','DETAILS OF GARDEN'],axis=1,inplace=True) #dropping the extra columns
df_ams.rename(columns={'g1':'GARDEN LOCATION'},inplace=True) #renaming the column
df_ams['GARDEN LOCATION'] = df_ams['GARDEN LOCATION'].str[1:-1]  #removing unnecessary characters from the string
df_ams['GARDEN LOCATION'] = df_ams['GARDEN LOCATION'].replace('',np.nan) #replacing unavailable data with NaN

# 2.LOCATION (#details like number similar to pincode can be extracted from this column)
df_ams[['LOCATION PIN','A','B','C','D']] = df_ams["LOCATION"].str.split(" ", expand = True) #splitting the column location with ' ' into 5 different columns
df_ams.drop(columns=['A','B','C','D','LOCATION'],axis=1,inplace=True) # dropping all other extra columns after keeping the location pin column

df_col_null = df_ams.columns[df_ams.isna().any()==True].tolist() #checking for columns with null values in the dataset
df_ams[df_col_null].isna().sum()
df_ams.drop(columns=['AREA OF GARDEN IN SQ METRE','GARDEN LOCATION'],axis=1,inplace=True) #removing columns with a large number of null values
df_ams['AVAILABILITY OF GARDEN'].fillna(df_ams['AVAILABILITY OF GARDEN'].mode()[0],inplace=True)  # filling null values with the mode of the column


# CATEGORICAL FEATURES

# 1.LABEL ENCODING (encode categorical features when there are more than 2 numer fo unique values in a column)
df_ams['UPKEEP STATUS'].unique()
df_ams['TYPE'].unique()
df_ams['INTERIOR'].unique()

le = LabelEncoder()  # encoding categories of features into numbers like 0,1,2,...
df_ams['INTERIOR']=le.fit_transform(df_ams['INTERIOR'])  #encoding  4 unique values in the column 'INTERIOR' to 0,1,2 and 3
df_ams['TYPE']=le.fit_transform(df_ams['TYPE'])  #encoding  3 unique values in the column 'TYPE' to 0,1 and 2
df_ams['UPKEEP STATUS']=le.fit_transform(df_ams['UPKEEP STATUS'])  #encoding  3 unique values in the column 'UPKEEP STATUS' into 0,1 and 2

#2. ONE HOT ENCODING (encode categorical features when there are only 2 unique values in a column)
df_ams['DETAILS OF BALCONY'].unique()
df_ams['GARAGE'].unique()
df_ams['DETAILS OF STORAGE'].unique()
df_ams['CONSTRUCTION TYPE'].unique()

dum_cols=df_ams[['DETAILS OF BALCONY','GARAGE','DETAILS OF STORAGE','CONSTRUCTION TYPE','AVAILABILITY OF GARDEN']] #List of columns to be encoded
dum_ams=pd.get_dummies(dum_cols)  # encoding categories into binary values
dum_ams.drop(columns=['DETAILS OF BALCONY_Not present','GARAGE_No','DETAILS OF STORAGE_Not present','CONSTRUCTION TYPE_Existing building','AVAILABILITY OF GARDEN_Not present'],axis=1,inplace=True)  #dropping extra columns
dum_ams.rename(columns={'DETAILS OF BALCONY_Present':'AVAILABILITY OF BALCONY','GARAGE_Yes':'AVAILABILITY OF GARAGE','DETAILS OF STORAGE_Present':'AVAILABILITY OF STORAGE','CONSTRUCTION TYPE_New development':'NEW BUILDING','AVAILABILITY OF GARDEN_Present':'AVAILABILITY OF GARDEN'},inplace=True)  #renaming columns
df_ams.drop(columns=['DETAILS OF BALCONY','GARAGE','DETAILS OF STORAGE','CONSTRUCTION TYPE','AVAILABILITY OF GARDEN'],axis=1,inplace=True)  # removing columns which have been used for encoding
df_ams=pd.concat((df_ams,dum_ams),axis=1)  #concatenating the encoded columns with the original dataframe

# converting datatypes as int
df_ams['NUMBER OF BEDROOMS'] = df_ams['NUMBER OF BEDROOMS'].astype(int)
df_ams['NUMBER OF BATHROOMS'] = df_ams['NUMBER OF BATHROOMS'].astype(int)
df_ams['NUMBER OF FLOORS'] = df_ams['NUMBER OF FLOORS'].astype(int)
df_ams['LOCATION PIN'] = df_ams['LOCATION PIN'].astype(int)

plt.figure(figsize=(10,5))  # heatmap for showing correlation between features
sns.heatmap(df_ams.corr(),annot=True)
plt.show()

df_ams.drop(columns=['TYPE',                    # dropping columns showing low correlation with price 
                     'AVAILABILITY OF BALCONY', # dropping 'NUMBER OF BEDROOMS' beacuse of multicollinearity   
                     'AVAILABILITY OF GARAGE',
                     'INTERIOR',
                     'NUMBER OF BEDROOMS',
                     'AVAILABILITY OF STORAGE',
                     'LOCATION PIN',
                     'AVAILABILITY OF GARDEN',
                     'NEW BUILDING',
                     'UPKEEP STATUS',
                     'CONSTRUCTION YEAR'],axis=1,inplace=True)
                     
df_ams.duplicated().sum()  # checking for duplicates and dropping them
df_ams.drop_duplicates(inplace=True)

# MODELLING

X=df_ams.drop(['PRICE PER MONTH'],axis=1) #independent features for making predictions
y=df_ams['PRICE PER MONTH']  #dependent feature

def detect_outliers(x):
    quartile_1, quartile_3 = np.percentile(df_ams, [25, 75]) # Set upper and lower bounds for finding outliers
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return (x > upper_bound) | (x < lower_bound) # Return a boolean mask indicating which values are outliers


outliers = detect_outliers(X)  # detecting outliers for X with independent features
X = X[~outliers]  # Remove outliers
X['VOLUME'].isna().sum()
X['VOLUME'].fillna(X['VOLUME'].median(),inplace=True)  #filling null values (which were created by removing outliers) with median of the values in the column

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)  #train-test split with 80% for training data and 20% for testing data

scaler = StandardScaler()  # Scaling the features for representing the values of all features in a same scale
scaler.fit(X_train)  # Fit the scaler to the training data
X_train_scaled = scaler.transform(X_train)  # Transform the training data
X_test_scaled = scaler.transform(X_test)  # Transform the training data


# LASSO REGRESSION

model = Lasso()  # Create a lasso regression model
param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}  # Create a dictionary of hyperparameters to search
grid_search = GridSearchCV(model, param_grid, cv=5) # Use GridSearchCV to tune the hyperparameters
grid_search.fit(X_train_scaled, y_train)  #fitting the lasso reression model
y_pred = grid_search.predict(X_test_scaled)  #predicting the price
print("Best hyperparameters:", grid_search.best_params_)  # Print the best hyperparameters
print("Mean Squared Error: %.3f" % mean_squared_error(y_test, y_pred))  # Evaluate the model on the test set
print("Test score: %.4f" % (r2_score(y_test,y_pred)))


# RIDGE REGRESSION

model = Ridge()  # Create a ridge regression model
param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}  # Create a dictionary of hyperparameters to search
grid_search = GridSearchCV(model, param_grid, cv=5)  # Use GridSearchCV to tune the hyperparameters
grid_search.fit(X_train_scaled, y_train)  #fitting the model for the tuned parameters
y_pred = grid_search.predict(X_test_scaled)  #predicting for the price
print("Best hyperparameters:", grid_search.best_params_)  # Print the best hyperparameters
print("Mean Squared Error: %.3f" % mean_squared_error(y_test, y_pred))  # Evaluate the model on the test set
print("Test Score: %.4f" %grid_search.score(X_test_scaled,y_test))


# RANDOM FOREST

model = RandomForestRegressor()  # Create a random forest model
param_distributions = {
    'n_estimators': [10, 100, 1000],   # Define the hyperparameter search space
    'max_depth': [None, 3, 5, 7],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
random_search = RandomizedSearchCV(model, param_distributions, n_iter=10, cv=5, return_train_score=True)  # Use RandomizedSearchCV to tune the hyperparameters
random_search.fit(X_train_scaled, y_train)  #fitting the model with the tuned parameters
y_pred = random_search.predict(X_test_scaled)  #predicting for the price
print("Best hyperparameters:", random_search.best_params_)  # Print the best hyperparameters
print("Mean Squared Error: %.3f" % mean_squared_error(y_test, y_pred))  # Evaluate the model on the test set
print("Test Score: %.3f" %random_search.score(X_test_scaled,y_test))


# XGBOOST REGRESSION

model = XGBRegressor()  # Create a XGBoost model
param_grid = {                # Define the hyperparameter grid
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.2, 0.3],
    'n_estimators': [100, 200, 300],
    'reg_lambda': [0.1, 1.0, 10.0]
}
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1)  # Use grid search to tune the hyperparameters
grid_search.fit(X_train_scaled, y_train)  #fitting the model
y_pred = grid_search.predict(X_test_scaled) #prediction done for the price
print(grid_search.best_params_)  # Print the best hyperparameters
print("Mean Squared Error: %.3f" % mean_squared_error(y_test, y_pred))  # Evaluate the model on the test set
print("Test Score: %.3f" %grid_search.score(X_test_scaled,y_test))

