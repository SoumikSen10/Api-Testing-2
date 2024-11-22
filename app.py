import os
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Model path
MODEL_PATH = '/opt/render/project/src/LCD.h5'

# Check if the model exists and load it
try:
    model = load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None  # Set the model to None if loading fails

@app.route('/')
def home():
    return "Welcome to the Breast Cancer Prediction App!"

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded correctly.'}), 500

    # Add prediction logic here
    # Example:
    try:
        # Your model prediction logic here (e.g., processing input image or data)
        prediction = "Your prediction result here"
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': f'Error during prediction: {e}'}), 500

# Make sure to bind to 0.0.0.0 and specify the port (5000 is standard for web services)
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Render provides the PORT environment variable
    app.run(host='0.0.0.0', port=port, debug=True)
