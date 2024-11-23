from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load the model
MODEL_PATH = './LCD.h5'
model = tf.keras.models.load_model(MODEL_PATH)

@app.route('/')
def home():
    return jsonify({'message': 'Lung Cancer Prediction API is running!'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure input is JSON
        data = request.get_json()

        if not data or 'features' not in data:
            return jsonify({'error': 'Missing "features" in request'}), 400

        # Extract features from the request
        features = np.array(data['features']).reshape(1, -1)  # Reshape for a single prediction

        # Model prediction
        prediction = model.predict(features)
        result = prediction[0][0]  # Assuming binary classification (e.g., 0 for No, 1 for Yes)

        return jsonify({'prediction': float(result)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
