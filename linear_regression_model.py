import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class LinearRegressionPredictor:
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = StandardScaler()
        
    def prepare_data(self, data, target_column):
        # Create features
        features = data.copy()
        if 'data' in features.columns:
            features = features.drop('data', axis=1)
        
        target = features.pop(target_column)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        return scaled_features, target
    
    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)
    
    def predict(self, X):
        predictions = self.model.predict(X)
        return predictions
    
    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2
        } 