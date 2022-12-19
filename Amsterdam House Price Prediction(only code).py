#!/usr/bin/env python
# coding: utf-8

# In[1]:


# AMSTERDAM HOUSE PRICE PREDICTION

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

df_ams=pd.read_csv('Amsterdam.csv')  

df_ams.head()
df_ams.info()
df_ams.dtypes
df_ams.columns
df_ams.describe()

df_ams.rename(columns = {'AREA IN m²':'AREA IN SQ METRE'}, inplace = True)   

df_ams.drop(columns=['TITLE',  
                     'File',
                     'URL',
                     'OFFERED SINCE',
                     'CONTACT DETAILS',
                     'TIMESTAMP',
                     'DESCRIPTION'],axis=1,inplace=True)

df_ams['VOLUME'] = df_ams['VOLUME'].replace('Volume is not available',np.nan)  
df_ams['INTERIOR'] = df_ams['INTERIOR'].replace('Interior is not available',np.nan)  
df_ams['AVAILABILITY'] = df_ams['AVAILABILITY'].replace('Not available to book',np.nan)  
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

df_col_null = df_ams.columns[df_ams.isna().any()==True].tolist()  

df_ams.drop(columns=['AVAILABILITY','SPECIFICATION','LOCATION TYPE','DESCRIPTION OF STORAGE'],axis=1,inplace=True)  

df_ams['VOLUME'] = df_ams['VOLUME'].astype(float) 

df_ams['VOLUME'].fillna(df_ams['VOLUME'].mean(),inplace=True) 
df_ams['GARAGE'].fillna(df_ams['GARAGE'].mode()[0],inplace=True)  
df_ams['INTERIOR'].fillna(df_ams['INTERIOR'].mode()[0],inplace=True)  
df_ams['UPKEEP STATUS'].fillna(df_ams['UPKEEP STATUS'].mode()[0],inplace=True)  
df_ams['NUMBER OF FLOORS'].fillna(df_ams['NUMBER OF FLOORS'].mode()[0],inplace=True) 
df_ams['DETAILS OF GARDEN'].fillna(df_ams['DETAILS OF GARDEN'].mode()[0],inplace=True) 
df_ams['DETAILS OF BALCONY'].fillna(df_ams['DETAILS OF BALCONY'].mode()[0],inplace=True) 
df_ams['DETAILS OF STORAGE'].fillna(df_ams['DETAILS OF STORAGE'].mode()[0],inplace=True)  
df_ams['NUMBER OF BEDROOMS'].fillna(df_ams['NUMBER OF BEDROOMS'].mode()[0],inplace=True)  
df_ams['NUMBER OF BATHROOMS'].fillna(df_ams['NUMBER OF BATHROOMS'].mode()[0],inplace=True)  

df_ams['NUMBER OF BEDROOMS'] = df_ams['NUMBER OF BEDROOMS'].astype(int)
df_ams['NUMBER OF BATHROOMS'] = df_ams['NUMBER OF BATHROOMS'].astype(int)
df_ams['NUMBER OF FLOORS'] = df_ams['NUMBER OF FLOORS'].astype(int)

# FEATURE ENGINEERING

# splitting the column with '(' into 2 different columns
df_ams[["AVAILABILITY OF GARDEN",'AREA OF GARDEN']] = df_ams["DETAILS OF GARDEN"].str.split("(", expand = True) 

#removing unnecessary characters from data
df_ams['AVAILABILITY OF GARDEN']=df_ams['AVAILABILITY OF GARDEN'].replace(['Present '],'Present')

#splitting the column 'Area of Garden' with 'm²' into 2 different columns of area and location
df_ams[['AREA OF GARDEN IN SQ METRE','GARDEN LOCATION']] = df_ams["AREA OF GARDEN"].str.split("m²", expand = True)

#removing extra characters and filling missing data with NaN
df_ams['GARDEN LOCATION'] = df_ams['GARDEN LOCATION'].str[1:-1]  
df_ams['GARDEN LOCATION'] = df_ams['GARDEN LOCATION'].replace('',np.nan)

#dropping the extra columns
df_ams.drop(columns=['AREA OF GARDEN','DETAILS OF GARDEN'],axis=1,inplace=True)

#splitting the column location with ' ' into 5 different columns
df_ams[['LOCATION PIN','A','B','C','D']] = df_ams["LOCATION"].str.split(" ", expand = True)

#dropping extra columns
df_ams.drop(columns=['A','B','C','D','LOCATION'],axis=1,inplace=True) 

#columns with null values
df_col_null = df_ams.columns[df_ams.isna().any()==True].tolist()
df_ams[df_col_null].isna().sum()

#drop columns with less data available
df_ams.drop(columns=['AREA OF GARDEN IN SQ METRE','GARDEN LOCATION'],axis=1,inplace=True)
df_ams['AVAILABILITY OF GARDEN'].fillna(df_ams['AVAILABILITY OF GARDEN'].mode()[0],inplace=True)


# CATEGORICAL FEATURES

df_ams['UPKEEP STATUS'].unique()
df_ams['TYPE'].unique()
df_ams['INTERIOR'].unique()

le = LabelEncoder() 
df_ams['INTERIOR']=le.fit_transform(df_ams['INTERIOR'])  
df_ams['TYPE']=le.fit_transform(df_ams['TYPE']) 
df_ams['UPKEEP STATUS']=le.fit_transform(df_ams['UPKEEP STATUS']) 

#2. ONE HOT ENCODING (encode categorical features when there are only 2 unique values in a column)

df_ams['DETAILS OF BALCONY'].unique()
df_ams['GARAGE'].unique()
df_ams['DETAILS OF STORAGE'].unique()
df_ams['CONSTRUCTION TYPE'].unique()

#dataframe with columns for which one hot encoding is needed
dum_cols=df_ams[['DETAILS OF BALCONY','GARAGE','DETAILS OF STORAGE','CONSTRUCTION TYPE','AVAILABILITY OF GARDEN']] 

#one-hot encoding
dum_ams=pd.get_dummies(dum_cols)  

#dropping extra columns and renaming the rest encoded columns
dum_ams.drop(columns=['DETAILS OF BALCONY_Not present',
                      'GARAGE_No',
                      'DETAILS OF STORAGE_Not present',
                      'CONSTRUCTION TYPE_Existing building',
                      'AVAILABILITY OF GARDEN_Not present'],axis=1,inplace=True) 
                      
dum_ams.rename(columns={'DETAILS OF BALCONY_Present':'AVAILABILITY OF BALCONY',
                        'GARAGE_Yes':'AVAILABILITY OF GARAGE',
                        'DETAILS OF STORAGE_Present':'AVAILABILITY OF STORAGE',
                        'CONSTRUCTION TYPE_New development':'NEW BUILDING',
                        'AVAILABILITY OF GARDEN_Present':'AVAILABILITY OF GARDEN'},inplace=True) 
                        
#dropped extra columns 
df_ams.drop(columns=['DETAILS OF BALCONY','GARAGE','DETAILS OF STORAGE','CONSTRUCTION TYPE','AVAILABILITY OF GARDEN'],axis=1,inplace=True)  

#concatenating encoded dataframe with original dataframe
df_ams=pd.concat((df_ams,dum_ams),axis=1) 

df_ams['LOCATION PIN'] = df_ams['LOCATION PIN'].astype(int)

plt.figure(figsize=(10,5))  
sns.heatmap(df_ams.corr(),annot=True)
plt.show()

df_ams.drop(columns=['TYPE',                   
                     'AVAILABILITY OF BALCONY',    
                     'AVAILABILITY OF GARAGE',
                     'INTERIOR',
                     'NUMBER OF BEDROOMS',
                     'AVAILABILITY OF STORAGE',
                     'LOCATION PIN',
                     'AVAILABILITY OF GARDEN',
                     'NEW BUILDING',
                     'UPKEEP STATUS',
                     'CONSTRUCTION YEAR'],axis=1,inplace=True)
                     
fi,axs=plt.subplots(nrows=2,ncols=3,figsize=(20,8))
cols=['AREA IN SQ METRE','NUMBER OF ROOMS','VOLUME','NUMBER OF BATHROOMS','NUMBER OF FLOORS']
for col,ax in zip(cols,axs.flat):
    sns.regplot(x=df_ams[col],y=df_ams['PRICE PER MONTH'],color='green',ax=ax)                     
                     
df_ams.drop_duplicates(inplace=True)

# MODELLING

X=df_ams.drop(['PRICE PER MONTH'],axis=1) 
y=df_ams['PRICE PER MONTH']  

def detect_outliers(x):
    quartile_1, quartile_3 = np.percentile(df_ams, [25, 75])
    iqr = quartile_3 - quartile_1
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return (x > upper_bound) | (x < lower_bound) 


outliers = detect_outliers(X) 
X = X[~outliers] 
X['VOLUME'].isna().sum()
X['VOLUME'].fillna(X['VOLUME'].median(),inplace=True)  

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42) 

scaler = StandardScaler() 
scaler.fit(X_train)  
X_train_scaled = scaler.transform(X_train)  
X_test_scaled = scaler.transform(X_test) 

# LASSO REGRESSION

model = Lasso()  

# Create a dictionary of hyperparameters to search
param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}  

# Use GridSearchCV to tune the hyperparameters
grid_search = GridSearchCV(model, param_grid, cv=5)

#fitting the lasso reression model
grid_search.fit(X_train_scaled, y_train) 

 #predicting the price 
y_pred = grid_search.predict(X_test_scaled) 

# Print the best hyperparameters
print("Best hyperparameters:", grid_search.best_params_)

 # Evaluate the model on the test set  
print("Mean Squared Error: %.3f" % mean_squared_error(y_test, y_pred)) 
print("Test score: %.4f" % (r2_score(y_test,y_pred)))


# RIDGE REGRESSION


model = Ridge() 
                                                
# Create a dictionary of hyperparameters to search
param_grid = {'alpha': [0.01, 0.1, 1, 10, 100]}  

# Use GridSearchCV to tune the hyperparameters               
grid_search = GridSearchCV(model, param_grid, cv=5)  

#fitting the model for the tuned parameters          
grid_search.fit(X_train_scaled, y_train)      

#predicting for the price                 
y_pred = grid_search.predict(X_test_scaled)       

# Print the best hyperparameters              
print("Best hyperparameters:", grid_search.best_params_)  

# Evaluate the model on the test set      
print("Mean Squared Error: %.3f" % mean_squared_error(y_test, y_pred))  
print("Test Score: %.4f" %grid_search.score(X_test_scaled,y_test))


# RANDOM FOREST

model = RandomForestRegressor()  

# Define the hyperparameter search space
param_distributions = {'n_estimators': [10, 100, 1000],   
                       'max_depth': [None, 3, 5, 7],
                       'min_samples_split': [2, 5, 10],
                       'min_samples_leaf': [1, 2, 4]}
                       
# Use RandomizedSearchCV to tune the hyperparameters
random_search = RandomizedSearchCV(model, param_distributions, n_iter=10, cv=5, return_train_score=True)  

#fitting the model with the tuned parameters
random_search.fit(X_train_scaled, y_train)  

 #predicting for the price
y_pred = random_search.predict(X_test_scaled) 

# Print the best hyperparameters
print("Best hyperparameters:", random_search.best_params_)  

# Evaluate the model 
print("Mean Squared Error: %.3f" % mean_squared_error(y_test, y_pred))  
print("Test Score: %.3f" %random_search.score(X_test_scaled,y_test))


# XGBOOST REGRESSION

model = XGBRegressor()  

# Define the hyperparameter grid
param_grid = {'max_depth': [3, 5, 7],
              'learning_rate': [0.1, 0.2, 0.3],
              'n_estimators': [100, 200, 300],
              'reg_lambda': [0.1, 1.0, 10.0]}
              
# Use grid search to tune the hyperparameters              
grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1)  

 #fitting the model
grid_search.fit(X_train_scaled, y_train) 

#prediction done for the price
y_pred = grid_search.predict(X_test_scaled) 

# Print the best hyperparameters
print(grid_search.best_params_)  

# Evaluate the model
print("Mean Squared Error: %.3f" % mean_squared_error(y_test, y_pred))  
print("Test Score: %.3f" %grid_search.score(X_test_scaled,y_test))

