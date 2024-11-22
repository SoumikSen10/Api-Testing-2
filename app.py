import os
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify

app = Flask(__name__)

# Set the path to the model, simulating the Render environment
MODEL_PATH = 'path/to/your/model/LCD.h5'  # Update this to the correct local path

@app.route('/')
def home():
    return "Lung Cancer Prediction API is running."

@app.route('/predict', methods=['POST'])
def predict():
    # Log to verify if the model exists locally
    print(f"Checking if model exists at {MODEL_PATH}...")
    if not os.path.exists(MODEL_PATH):
        return jsonify({'error': f'Model file not found at {MODEL_PATH}'}), 500

    try:
        model = load_model(MODEL_PATH)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        return jsonify({'error': 'Error loading model.'}), 500

    # Prediction logic here...
    return jsonify({'message': 'Model loaded and ready for predictions.'})

if __name__ == '__main__':
    app.run(debug=True)
