import os
import json
import tensorflow as tf
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Global variable to store model
model = None

# Set memory growth for GPU (if available)
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(f"Error setting memory growth: {e}")

# Function to load the model lazily
def get_model():
    global model
    if model is None:
        print("Loading model...")
        model = load_model('/opt/render/project/src/LCD.h5', compile=False)  # Update path as needed
    return model

# Example data preprocessing function (adjust as per your input requirements)
def preprocess_data(input_data):
    # Convert the input to numpy array or any format your model expects
    data = np.array(input_data)
    # Any other preprocessing steps here (e.g., normalization)
    return data

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data
        input_data = request.get_json()  # Expecting JSON body
        if not input_data:
            return jsonify({"error": "No input data provided"}), 400

        # Preprocess data
        data = preprocess_data(input_data)

        # Load model if not already loaded
        model = get_model()

        # Make prediction
        predictions = model.predict(data)
        
        # Format the prediction result as needed
        result = predictions.tolist()  # Convert to list for JSON serialization
        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "OK"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
