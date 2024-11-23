import os

# Disable GPU usage (use CPU only)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
from flask import Flask, jsonify, request
import numpy as np
import json
import os

app = Flask(__name__)

# Load the model once at the start of the application
model_file_path = '/opt/render/project/src/LCD.h5'

# Debugging: Check if model file exists and is accessible
if os.path.exists(model_file_path):
    print(f"Model file exists: {model_file_path}")
else:
    print(f"Model file not found at: {model_file_path}")

# Check if we can load the model successfully
try:
    print(f"Loading model from: {model_file_path}")
    model = load_model(model_file_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

@app.route('/')
def index():
    return "Model API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure request contains file
        if 'file' not in request.files:
            return jsonify({'error': 'No file part in the request'}), 400

        file = request.files['file']

        # Ensure file is not empty
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Read and process the file (assuming it's an image)
        image = np.array(file.read())  # This is just an example, you might need to process it as per your model's requirement
        
        # Assuming the model expects a 4D tensor of shape (batch_size, height, width, channels)
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        
        # Make prediction
        prediction = model.predict(image)

        # Return prediction as JSON response
        return jsonify({'prediction': prediction.tolist()})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Prediction error occurred'}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
