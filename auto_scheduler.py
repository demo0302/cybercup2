import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
import joblib
import datetime
import json

class CloudResourcePredictor:
    def __init__(self):
        self.cpu_model = None
        self.memory_model = None
        self.scaler = StandardScaler()
        
    def prepare_features(self, data):
        """Convert timestamp to hour and day features"""
        df = pd.DataFrame(data)
        df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
        df['day_of_week'] = pd.to_datetime(df['timestamp']).dt.dayofweek
        return df[['hour', 'day_of_week', 'user_count']]
    
    def train(self, historical_data):
        """Train the model on historical data"""
        X = self.prepare_features(historical_data)
        y_cpu = historical_data['cpu_utilization']
        y_memory = historical_data['memory_utilization']
        
        # Train CPU model using CNN
        X_train, X_test, y_train, y_test = train_test_split(X, y_cpu, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.cpu_model = Sequential()
        self.cpu_model.add(Conv1D(64, kernel_size=2, activation='relu', input_shape=(X_train_scaled.shape[1], 1)))
        self.cpu_model.add(MaxPooling1D(pool_size=2))
        self.cpu_model.add(Flatten())
        self.cpu_model.add(Dense(50, activation='relu'))
        self.cpu_model.add(Dense(1))
        self.cpu_model.compile(optimizer='adam', loss='mean_squared_error')
        self.cpu_model.fit(X_train_scaled, y_train, epochs=10, validation_data=(X_test_scaled, y_test), batch_size=10)
        
        # Train Memory model using CNN
        X_train, X_test, y_train, y_test = train_test_split(X, y_memory, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.memory_model = Sequential()
        self.memory_model.add(Conv1D(64, kernel_size=2, activation='relu', input_shape=(X_train_scaled.shape[1], 1)))
        self.memory_model.add(MaxPooling1D(pool_size=2))
        self.memory_model.add(Flatten())
        self.memory_model.add(Dense(50, activation='relu'))
        self.memory_model.add(Dense(1))
        self.memory_model.compile(optimizer='adam', loss='mean_squared_error')
        self.memory_model.fit(X_train_scaled, y_train, epochs=10, validation_data=(X_test_scaled, y_test), batch_size=10)
    
    def predict(self, current_data):
        """Predict CPU and memory utilization"""
        X = self.prepare_features(current_data)
        X_scaled = self.scaler.transform(X)
        X_scaled = np.expand_dims(X_scaled, axis=2)
        cpu_pred = self.cpu_model.predict(X_scaled)[0][0]
        memory_pred = self.memory_model.predict(X_scaled)[0][0]
        return cpu_pred, memory_pred
        
    def save_models(self):
        """Save trained models"""
        self.cpu_model.save('cpu_model.h5')
        self.memory_model.save('memory_model.h5')
        joblib.dump(self.scaler, 'scaler.joblib')
    
    def load_models(self):
        """Load trained models"""
        self.cpu_model = load_model('cpu_model.h5')
        self.memory_model = load_model('memory_model.h5')
        self.scaler = joblib.load('scaler.joblib')

class AutoScaler:
    def __init__(self, predictor, cpu_threshold=70, memory_threshold=70):
        self.predictor = predictor
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        
    def get_scaling_recommendation(self, current_data):
        """Determine if scaling is needed based on predictions"""
        cpu_pred, memory_pred = self.predictor.predict(current_data)
        
        if cpu_pred > self.cpu_threshold or memory_pred > self.memory_threshold:
            return "scale_up"
        elif cpu_pred < self.cpu_threshold/2 and memory_pred < self.memory_threshold/2:
            return "scale_down"
        return "no_action"

# Example usage function
def simulate_auto_scaling():
    # Generate sample historical data
    timestamps = pd.date_range(start='2024-01-01', end='2024-01-31', freq='H')
    n_samples = len(timestamps)
    
    historical_data = {
        'timestamp': timestamps,
        'user_count': np.random.randint(10, 100, n_samples),
        'cpu_utilization': np.random.uniform(20, 90, n_samples),
        'memory_utilization': np.random.uniform(20, 90, n_samples)
    }
    
    # Create and train the predictor
    predictor = CloudResourcePredictor()
    predictor.train(historical_data)
    
    # Create auto-scaler
    auto_scaler = AutoScaler(predictor)
    
    # Simulate current data
    current_data = {
        'timestamp': [pd.Timestamp.now()],
        'user_count': [50]
    }
    
    # Get scaling recommendation
    recommendation = auto_scaler.get_scaling_recommendation(current_data)
    
    return recommendation

if __name__ == "__main__":
    recommendation = simulate_auto_scaling()
    print(f"Scaling recommendation: {recommendation}")
