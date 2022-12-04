import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
#Clase para manejo de variables temporales en el modelo de House Price
class TremporalVariableTransformer(BaseEstimator, TransformerMixin):
    #Constructor
    def __init__(self, variables, reference_variable):
        if not isinstance(variables, list):
            raise ValueError("Las varibles debe ser incluida en una lista.")
        self.variables = variables
        self.reference_variable = reference_variable
        
    #metodo fit para habilitar metodo transform
    def fit(self, X, y=None):
        return self
    #metodo para transformar variables temporales.
    def transform(self, X):
        X = X.copy()
        for feature in self.variables:
            X[feature] = X[self.reference_variable] - X[feature]
        return X
    
#Clase para transformación de variables categóricas ordinales.
class Mapper(BaseEstimator, TransformerMixin):
    #constructor
    def __init__(self, variables, mappings):
        if not isinstance(variables, list):
            raise ValueError("Las varibles debe ser incluida en una lista.")
        #campos de clase Mapper.
        self.variables = variables
        self.mappings = mappings
        
    #Metodo Fit
    def fit(self, X, y=None):
        #fit no hace nada, pero es requisito para el pipeline
        return self
    
    #Metodo transform
    def transform(self, X):
        X = X.copy()
        for variable in self.variables:
            X[variable] = X[variable].map(self.mappings)
        return X