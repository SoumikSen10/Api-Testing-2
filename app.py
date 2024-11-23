import os
import sys
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import contextlib
import tensorflow as tf
import time

# Suppress TensorFlow INFO and WARNING logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')

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

# Load the model lazily (only once)
def get_model():
    global model
    if model is None:
        try:
            print("Loading model...")
            model = load_model('LCD.h5', compile=False)  # Update path as needed
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            sys.exit(1)
    return model

# Image preprocessing function
def load_and_preprocess_image(img_path, target_size):
    try:
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the image
        return img_array
    except Exception as e:
        print(f"Error in loading and preprocessing image: {e}")
        sys.exit(1)

# Prediction function
def predict_image_class(model, img_path, target_size):
    try:
        img = load_and_preprocess_image(img_path, target_size)
        
        # Suppress stdout/stderr from TensorFlow during prediction
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            predictions = model.predict(img)
        
        # Class labels for the model
        class_labels = ['squamous cell carcinoma', 'large cell carcinoma', 'normal', 'adenocarcinoma']
        
        predicted_class = np.argmax(predictions[0])  # Get the class with the highest probability
        predicted_label = class_labels[predicted_class]
        
        # Return 'cancerous' or 'non-cancerous' based on the prediction
        if predicted_label == 'normal':
            return 'non-cancerous'
        else:
            return 'cancerous'
    except Exception as e:
        print(f"Error in predicting image class: {e}")
        sys.exit(1)

# Health check endpoint
@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "OK"}), 200

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    start_time = time.time()

    try:
        # Check if a file is part of the request
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        
        # Ensure file is provided
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        # Save the image to a temporary path
        img_path = os.path.join("uploads", file.filename)
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        file.save(img_path)
        
        print(f"File saved at: {img_path}")  # Debugging line
        
        # Load model
        model = get_model()
        
        # Image size expected by the model
        IMAGE_SIZE = (256, 256)
        
        # Predict the class (cancerous or non-cancerous)
        result = predict_image_class(model, img_path, IMAGE_SIZE)
        
        # Measure time taken for prediction
        prediction_time = time.time() - start_time
        print(f"Prediction took {prediction_time:.2f} seconds")

        return jsonify({"prediction": result})

    except Exception as e:
        print(f"Error during prediction: {str(e)}")  # Debugging line
        return jsonify({"error": str(e)}), 500

# Run the Flask app with port binding for cloud platforms
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
