from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import sys
import contextlib
import tensorflow as tf
from PIL import Image

# Suppress TensorFlow INFO and WARNING logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
tf.get_logger().setLevel('ERROR')

# Initialize Flask app
app = Flask(__name__)

# Model loading
model_path = os.path.join(os.path.dirname(__file__), 'LCD.h5')
if not os.path.exists(model_path):
    print(f"Model file not found at {model_path}")
    sys.exit(1)

try:
    model = load_model(model_path, compile=False)  # Prevent issues with older models
except Exception as e:
    print(f"Error loading model: {e}")
    sys.exit(1)

# Image preprocessing function
IMAGE_SIZE = (256, 256)
class_labels = ['squamous cell carcinoma', 'large cell carcinoma', 'normal', 'adenocarcinoma']

def load_and_preprocess_image(img):
    try:
        # Convert image to RGB (removes alpha channel if it exists)
        img = img.convert('RGB')
        img = img.resize(IMAGE_SIZE)  # Resize the image to 256x256
        img_array = np.array(img)
        img_array = img_array.astype('float32')  # Convert to float32
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the image
        return img_array
    except Exception as e:
        print(f"Error in loading and preprocessing image: {e}")
        return None

# Prediction function
def predict_image_class(img):
    try:
        img_array = load_and_preprocess_image(img)
        if img_array is None:
            return None

        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            predictions = model.predict(img_array)
        
        predicted_class = np.argmax(predictions[0])
        predicted_label = class_labels[predicted_class]
        
        if predicted_label == 'normal':
            return 'non-cancerous'
        else:
            return 'cancerous'
    except Exception as e:
        print(f"Error in predicting image class: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict():
    # Check if an image is part of the request
    if 'image' not in request.files:
        return jsonify({"error": "No image in request.files"}), 400

    image_file = request.files['image']
    
    if image_file.filename == '':
        return jsonify({"error": "No selected image"}), 400

    try:
        # Open the image using PIL
        img = Image.open(image_file.stream)
        result = predict_image_class(img)
        
        if result is None:
            return jsonify({"error": "Error in prediction process"}), 500
        
        return jsonify({"prediction": result}), 200
    
    except Exception as e:
        print(f"Error in processing image: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

