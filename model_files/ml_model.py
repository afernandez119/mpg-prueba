import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer



def preprocess(data):
    
    data["origin"] = data["origin"].map({1: 'India', 2: 'USA', 3 : 'Germany'})
    
    return data

class CustColAdder(BaseEstimator, TransformerMixin):
    '''
    Clase específica para añadir nuevas columnas al DF en un PipeLine. Es importante que las columnas estén en la posición
    indicada a continuación:
        acceleration_ix = 4
        cylinders_ix = 0
        horsepower_ix = 2
        weight_ix = 3
    '''
        
    #def __init__(self):
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        acceleration_ix = 4 #data.columns.get_loc("acceleration")
        cylinders_ix = 0 #data.columns.get_loc("cylinders")
        horsepower_ix = 2 #data.columns.get_loc("horsepower")
        weight_ix = 3
        
        acc_on_cyl = X[:, acceleration_ix] / X[:, cylinders_ix] # required new variable
        acc_on_weight = X[:, acceleration_ix] / X[:, weight_ix]  
        horse_on_acc = X[:, horsepower_ix] / X[:, acceleration_ix]  
        
        return np.c_[X, acc_on_cyl, acc_on_weight, horse_on_acc]
    
    
    
def num_pipeline_transformer(data):
    '''
    Function to process numerical transformations
    Argument:
        data: original dataframe 
    Returns:
        num_attrs: numerical dataframe
        num_pipeline: numerical pipeline object
    '''
    
    numerics = ['float64', 'int64']

    num_attrs = data.select_dtypes(include=numerics) #devuelve df con columnas de los datos seleccionados
       
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('attrs_adder', CustColAdder()),
        ('std_scaler', StandardScaler()),
        ])
    return num_attrs, num_pipeline


def pipeline_transformer(data):
    '''
    Complete transformation pipeline for both
    nuerical and categorical data.
    
    Argument:
        data: original dataframe 
    Returns:
        prepared_data: transformed data, ready to use
    '''
    cat_attrs = ["origin"]
    
    num_attrs, num_pipeline = num_pipeline_transformer(data)
    
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, list(num_attrs)),
        ("cat", OneHotEncoder(), cat_attrs),
        ])
    
    prepared_data = full_pipeline.fit_transform(data)
    return prepared_data


def predict_mpg(model, data):
    data_processed = pd.DataFrame(data)
    data_processed = preprocess(data_processed)
    data_processed = pipeline_transformer(data_processed)
    results = model.predict(data_processed)

    return results