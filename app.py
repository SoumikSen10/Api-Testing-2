from flask import Flask, request, jsonify
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)

model = None

IMAGE_SIZE = (256, 256)
class_labels = ['squamous cell carcinoma', 'large cell carcinoma', 'normal', 'adenocarcinoma']

def load_model_lazy():
    global model
    if model is None:
        print("Loading model...")
        model = load_model('/opt/render/project/src/LCD.h5', compile=False)
    return model

def load_and_preprocess_image(img_path, target_size):
    try:
        img = image.load_img(img_path, target_size=target_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        return img_array
    except Exception as e:
        print(f"Error in loading and preprocessing image: {e}")
        return None

def predict_image_class(model, img_path, target_size):
    try:
        img = load_and_preprocess_image(img_path, target_size)
        if img is None:
            return None
        predictions = model.predict(img)
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
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['file']

        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({"error": "Invalid file format. Only PNG, JPG, JPEG allowed."}), 415

        img_path = os.path.join('/tmp', file.filename)
        file.save(img_path)

        model = load_model_lazy()
        if model is None:
            return jsonify({"error": "Model could not be loaded"}), 500

        prediction = predict_image_class(model, img_path, IMAGE_SIZE)

        if not prediction:
            return jsonify({"error": "Error in image prediction"}), 500

        return jsonify({"prediction": prediction})

    except Exception as e:
        print(f"Error in the request: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "OK"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
