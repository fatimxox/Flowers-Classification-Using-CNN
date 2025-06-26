import os
import numpy as np
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import cv2
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import logging
import random
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
MODEL_PATH = 'model.h5'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Class names for your model
CLASS_NAMES = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']

# Load the pre-trained model
model = None
try:
    logger.info("Loading model from %s...", MODEL_PATH)
    model = load_model(MODEL_PATH)
    logger.info("Model loaded successfully")
    logger.info("Model input shape: %s", model.input_shape)
    logger.info("Model output shape: %s", model.output_shape)
except Exception as e:
    logger.error("Error loading model: %s", str(e), exc_info=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def prepare_image(uploaded_file):
    """Prepare the uploaded image for EfficientNetB3 prediction."""
    try:
        # Read image
        img = cv2.imread(uploaded_file)
        if img is None:
            raise ValueError(f"Failed to load image from {uploaded_file}")
        logger.info("Original image shape: %s", img.shape)
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Add slight random variations for prediction diversity
        # Random brightness adjustment (smaller range for EfficientNet)
        brightness = random.uniform(0.95, 1.05)
        img = cv2.multiply(img, brightness)
        
        # Random rotation (smaller angle for EfficientNet)
        angle = random.uniform(-5, 5)
        height, width = img.shape[:2]
        matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1.0)
        img = cv2.warpAffine(img, matrix, (width, height))
        
        # Resize to 224x224 (EfficientNetB3 input size)
        img = cv2.resize(img, (224, 224))
        
        # Convert to float32
        img = img.astype(np.float32)
        
        # Apply EfficientNet preprocessing
        img = preprocess_input(img)
        
        # Add batch dimension
        img_input = np.expand_dims(img, axis=0)
        logger.info("Final input shape: %s", img_input.shape)
        
        return img_input
        
    except Exception as e:
        logger.error("Error in prepare_image: %s", str(e), exc_info=True)
        raise

def process_predictions(raw_predictions, temperature=1.0):
    """Process predictions with temperature scaling."""
    try:
        logger.info("Raw predictions shape: %s", raw_predictions.shape)
        logger.info("Raw prediction values: %s", raw_predictions)
        
        # Since your model already has a softmax layer, we don't need to apply it again
        predictions = raw_predictions
        
        # Add very small random noise to break potential ties
        noise = np.random.normal(0, 0.0001, predictions.shape)
        predictions = predictions + noise
        
        # Renormalize
        predictions = predictions / np.sum(predictions, axis=1, keepdims=True)
        
        # Log prediction distribution
        for i, pred in enumerate(predictions[0]):
            logger.info("Class %s: %.4f", CLASS_NAMES[i], pred)
            
        return predictions
        
    except Exception as e:
        logger.error("Error in process_predictions: %s", str(e), exc_info=True)
        raise

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        logger.info("Received POST request")
        
        if model is None:
            logger.error("Model not loaded")
            return jsonify({'error': 'Model not loaded correctly'})
        
        if 'file' not in request.files:
            logger.warning("No file in request")
            return jsonify({'error': 'No file uploaded'})
            
        file = request.files['file']
        
        if file.filename == '':
            logger.warning("Empty filename")
            return jsonify({'error': 'No selected file'})
            
        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                logger.info("Saving file to %s", filepath)
                file.save(filepath)
                
                # Process image
                logger.info("Processing image...")
                processed_image = prepare_image(filepath)
                
                # Make prediction
                logger.info("Making prediction...")
                predictions = model.predict(processed_image, verbose=0)
                
                # Process predictions (minimal processing since model has softmax)
                predictions = process_predictions(predictions)
                
                # Create response
                top_predictions = []
                for i, confidence in enumerate(predictions[0]):
                    if i < len(CLASS_NAMES):
                        top_predictions.append({
                            'class': CLASS_NAMES[i],
                            'confidence': float(confidence)
                        })
                
                top_predictions.sort(key=lambda x: x['confidence'], reverse=True)
                logger.info("Final top predictions: %s", top_predictions[:3])
                
                return jsonify({
                    'filename': filename,
                    'predictions': top_predictions[:3]
                })
                
            except Exception as e:
                logger.error("Error processing request: %s", str(e), exc_info=True)
                return jsonify({'error': str(e)})
        
        return jsonify({'error': 'File type not allowed'})
    
    return render_template('index.html')

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)