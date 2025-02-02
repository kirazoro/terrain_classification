﻿# **Terrain Classifier Web App**

This is a Flask-based web application that uses a TensorFlow/Keras model to classify terrains (e.g., grassy, marshy, rocky, sandy) from captured images. The app allows users to capture images through their device camera and get real-time predictions about the terrain type.

---

## **Features**
- Capture images directly using your device's camera.
- Classify images into terrain types using a pre-trained TensorFlow/Keras model.
- Display predictions with confidence percentages.
- User-friendly interface with a live video feed for capturing images.

---

## **Folder Structure**
```
terrain-classifier/
├── app.py                        # Flask application code
├── captired_images                  # Folder to store uploaded images
├── templates/
│   ├── index.html                # Homepage template
│   └── result.html               # Classification result template
├── terrain_classification_model.h5  # Pre-trained Keras model
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

---

## **Setup and Installation**

### **Prerequisites**
- Python 3.8 or above
- Flask
- TensorFlow/Keras
- PIL (Pillow)

### **Steps to Run the Project**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repository/terrain-classifier.git
   cd terrain-classifier
   ```

2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your pre-trained model (`terrain_classification_model.h5`) in the root directory.

4. Run the Flask application:
   ```bash
   python app.py
   ```

5. Open your web browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

---

## **Usage**
1. Access the homepage to see the live video feed.
2. Click the "Capture" button to take a photo.
3. The app will process the image, classify the terrain, and display the result with confidence percentage.

---

## **Technologies Used**
- **Frontend**: HTML, CSS, JavaScript
- **Backend**: Flask (Python)
- **Machine Learning**: TensorFlow/Keras for terrain classification
- **Image Processing**: PIL (Pillow)

---

## **Example Output**
- Predicted Terrain: `Grassy`
- Confidence: `98.7%`
- Display of the uploaded image alongside the prediction.

---

## **Contributing**
1. Fork the repository.
2. Create a new branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add feature description"
   ```
4. Push to the branch:
   ```bash
   git push origin feature-name
   ```
5. Submit a pull request.

---

## **Acknowledgements**
- [TensorFlow](https://www.tensorflow.org/) for model training and inference.
- Flask for web application framework.
- Community contributions and tutorials.
