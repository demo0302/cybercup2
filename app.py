from flask import Flask, jsonify, request
import numpy as np
import pandas as pd
import tensorflow as tf
from auto_scheduler import CloudResourcePredictor, AutoScaler

app = Flask(__name__)

# Load the pre-trained LSTM model
lstm_model = tf.keras.models.load_model('saved_model/lstm_model.h5')

# Load the pre-trained CNN model
cnn_model = tf.keras.models.load_model('saved_model/cnn_model.h5')  # Make sure to train and save the CNN model

SEQ_LENGTH = 20

# Initialize the CloudResourcePredictor and AutoScaler
predictor = CloudResourcePredictor()
predictor.load_models()
auto_scaler = AutoScaler(predictor)

# Function to simulate real-time data
def simulate_real_time_data():
    timestamp = [pd.Timestamp.now()]
    user_count = [50]  # Example user count
    return {'timestamp': timestamp, 'user_count': user_count}

@app.route('/predict_lstm', methods=['GET'])
def predict_lstm():
    current_data = simulate_real_time_data()
    cpu_pred, memory_pred = lstm_predict(current_data)
    return jsonify({
        'cpu_prediction': cpu_pred,
        'memory_prediction': memory_pred
    })

@app.route('/predict_cnn', methods=['GET'])
def predict_cnn():
    current_data = simulate_real_time_data()
    cpu_pred, memory_pred = cnn_predict(current_data)
    return jsonify({
        'cpu_prediction': cpu_pred,
        'memory_prediction': memory_pred
    })

@app.route('/predict_and_scale', methods=['GET'])
def predict_and_scale():
    current_data = simulate_real_time_data()
    cpu_pred, memory_pred = lstm_predict(current_data)
    actions = auto_scaler.get_scaling_recommendation(current_data)
    return jsonify({
        'cpu_prediction': cpu_pred,
        'memory_prediction': memory_pred,
        'actions': actions
    })

def lstm_predict(data):
    data_scaled = preprocess_data(data)
    prediction = lstm_model.predict(data_scaled)
    cpu_pred = prediction[0][0] * 100  # Dummy scaling
    memory_pred = prediction[0][1] * 100  # Dummy scaling
    return cpu_pred, memory_pred

def cnn_predict(data):
    data_scaled = preprocess_data(data)
    prediction = cnn_model.predict(data_scaled)
    cpu_pred = prediction[0][0] * 100  # Dummy scaling
    memory_pred = prediction[0][1] * 100  # Dummy scaling
    return cpu_pred, memory_pred

def preprocess_data(data):
    # Implement your preprocessing logic here
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data[['cpu_usage', 'memory_usage']])
    return np.array(data_scaled).reshape(1, SEQ_LENGTH, 2)

if __name__ == '__main__':
    app.run(debug=True)
