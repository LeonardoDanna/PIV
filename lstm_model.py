import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class LSTMPredictor:
    def __init__(self, sequence_length=7):
        self.sequence_length = sequence_length
        self.model = None
        self.scaler = MinMaxScaler()
        
    def prepare_data(self, data, target_column):
        # Create a copy of the data and drop the date column
        data_copy = data.copy()
        if 'data' in data_copy.columns:
            data_copy = data_copy.drop('data', axis=1)
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(data_copy)
        
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:(i + self.sequence_length)])
            y.append(scaled_data[i + self.sequence_length, data_copy.columns.get_loc(target_column)])
            
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape):
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def train(self, X_train, y_train, epochs=50, batch_size=32):
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=0
        )
        return history
    
    def predict(self, X):
        predictions = self.model.predict(X)
        # Inverse transform predictions
        dummy_array = np.zeros((len(predictions), X.shape[2]))
        dummy_array[:, 0] = predictions.flatten()
        predictions = self.scaler.inverse_transform(dummy_array)[:, 0]
        return predictions
    
    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        # Inverse transform y_test
        dummy_array = np.zeros((len(y_test), X_test.shape[2]))
        dummy_array[:, 0] = y_test
        y_test_original = self.scaler.inverse_transform(dummy_array)[:, 0]
        
        mse = mean_squared_error(y_test_original, predictions)
        mae = mean_absolute_error(y_test_original, predictions)
        r2 = r2_score(y_test_original, predictions)
        
        return {
            'mse': mse,
            'mae': mae,
            'r2': r2
        } 