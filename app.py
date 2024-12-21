from flask import Flask, render_template, request, redirect, url_for
import cv2
import os
import tensorflow as tf
import numpy as np
from PIL import Image
import uuid

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = "captured_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Constants
MODEL_PATH = "terrain_classifier.h5"  # Path to the TensorFlow/Keras model
IMG_SIZE = (224, 224)  # Image size expected by the model
CLASS_NAMES = ["grassy", "marshy", "rocky", "sandy"]  # Class labels

# Load the TensorFlow model
model = tf.keras.models.load_model(MODEL_PATH)

# Function to preprocess the image
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Ensure it's RGB
    image = image.resize(IMG_SIZE)  # Resize to model input size
    input_data = np.array(image, dtype=np.float32) / 255.0  # Normalize
    input_data = np.expand_dims(input_data, axis=0)  # Add batch dimension
    return input_data

# Function to classify the image
def classify_image(image_path):
    # Preprocess the image
    input_data = preprocess_image(image_path)
    
    # Run inference
    predictions = model.predict(input_data)
    predicted_class = np.argmax(predictions)
    
    # Return the class name and confidence
    confidence = predictions[0][predicted_class] * 100
    return CLASS_NAMES[predicted_class], confidence

# Route to capture an image
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/capture", methods=["POST"])
def capture_image():
    # Open webcam and capture the image
    camera = cv2.VideoCapture(0)
    ret, frame = camera.read()
    if not ret:
        return "Failed to capture image."
    camera.release()

    # Save the captured image
    file_name = f"{uuid.uuid4().hex}.jpg"
    file_path = os.path.join(UPLOAD_FOLDER, file_name)
    cv2.imwrite(file_path, frame)

    # Classify the captured image
    terrain_type, confidence = classify_image(file_path)

    return render_template(
        "result.html",
        terrain_type=terrain_type,
        confidence=confidence,
        image_path=file_path,
    )

if __name__ == "__main__":
    app.run(debug=True)
