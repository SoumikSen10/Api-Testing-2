import os
import sys
import numpy as np
import contextlib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify
import tensorflow as tf

# Initialize Flask app
app = Flask(__name__)

# Suppress TensorFlow INFO and WARNING logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')

# Global model variable
model = None

# Class labels for cancer types
class_labels = ['squamous cell carcinoma', 'large cell carcinoma', 'normal', 'adenocarcinoma']

# Path to model file (update as needed)
model_path = '/opt/render/project/src/LCD.h5'  # Update with the correct path

# Image size expected by the model
IMAGE_SIZE = (256, 256)

# Load model lazily when needed
def load_model_lazy():
    global model
    if model is None:
        try:
            print("Loading model...")
            model = load_model(model_path, compile=False)  # Load model without compiling
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    return model

# Function to preprocess image
def load_and_preprocess_image(img_path, target_size):
    try:
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize image
        return img_array
    except Exception as e:
        print(f"Error in loading and preprocessing image: {e}")
        return None

# Prediction logic (image classification)
def predict_image_class(model, img_path, target_size):
    try:
        img = load_and_preprocess_image(img_path, target_size)
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            predictions = model.predict(img)
        predicted_class = np.argmax(predictions[0])
        predicted_label = class_labels[predicted_class]
        return 'non-cancerous' if predicted_label == 'normal' else 'cancerous'
    except Exception as e:
        print(f"Error in predicting image class: {e}")
        return None

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Ensure file is present in the request
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400

        file = request.files['file']
        
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Ensure file format is allowed (only images)
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({"error": "Invalid file format. Only PNG, JPG, JPEG allowed."}), 415

        # Save file temporarily
        img_path = os.path.join('/tmp', file.filename)
        file.save(img_path)

        # Load the model if not already loaded
        model = load_model_lazy()
        if model is None:
            return jsonify({"error": "Model could not be loaded"}), 500

        # Make prediction
        prediction = predict_image_class(model, img_path, IMAGE_SIZE)
        
        if not prediction:
            return jsonify({"error": "Error in image prediction"}), 500

        # Return the result
        return jsonify({"prediction": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "OK"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
