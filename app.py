import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify

app = Flask(__name__)

# Path to the model
MODEL_PATH = '/opt/render/project/src/LCD.h5'

# Load the model when the app starts
def load_lung_cancer_model():
    try:
        # Check if the model file exists at the expected location
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")
        model = load_model(MODEL_PATH)
        print("Model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

model = load_lung_cancer_model()

@app.route('/')
def home():
    return "Lung Cancer Prediction API is running."

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded correctly.'}), 500

    try:
        # Ensure a file is uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided.'}), 400

        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected.'}), 400

        # Save the uploaded file temporarily
        file_path = os.path.join('/tmp', file.filename)
        file.save(file_path)

        # Preprocess the image (adjust as needed)
        # This is assuming you are using image data
        img = tf.keras.preprocessing.image.load_img(file_path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Add batch dimension

        # Predict with the loaded model
        predictions = model.predict(img_array)
        
        # Assuming the output is a binary classification (0 or 1)
        predicted_class = 'Cancer' if predictions[0] > 0.5 else 'No Cancer'

        # Clean up the file after prediction
        os.remove(file_path)

        return jsonify({'prediction': predicted_class})

    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({'error': 'Error processing the request.'}), 500

if __name__ == '__main__':
    # Run the Flask app on port 5000 (for local testing, adjust if needed)
    app.run(debug=True, host='0.0.0.0', port=5000)
