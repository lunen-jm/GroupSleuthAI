import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2

# Define the emotion labels
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def load_emotion_model(model_path='models/emotion_recognition_transfer_precision.h5'):
    """
    Load the pre-trained emotion detection model
    
    Args:
        model_path: Path to the model file
        
    Returns:
        Loaded TensorFlow model
    """
    try:
        # Handle relative paths
        if not os.path.isabs(model_path):
            # Try different relative paths to find the model
            possible_paths = [
                model_path,
                os.path.join(os.getcwd(), model_path),
                os.path.join(os.path.dirname(os.path.dirname(__file__)), model_path)
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    break
        
        print(f"Loading emotion model from: {model_path}")
        model = load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading emotion model: {e}")
        print("Using dummy model instead")
        return create_dummy_emotion_model()

def create_dummy_emotion_model():
    """Create a simple dummy model for testing when the real model isn't available"""
    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    outputs = tf.keras.layers.Dense(7, activation='softmax')(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

def preprocess_face_for_emotion(face_img, target_size=(224, 224)):
    """
    Preprocess a face image for emotion detection
    
    Args:
        face_img: Face image (RGB)
        target_size: Target size for the model
        
    Returns:
        Preprocessed face ready for the model
    """
    if face_img.size == 0:
        raise ValueError("Empty face image provided")
    
    # Resize to target size
    face_img = cv2.resize(face_img, target_size)
    
    # Convert to array and add batch dimension
    face_array = np.array(face_img)
    
    # Apply MobileNetV2 preprocessing
    face_array = preprocess_input(face_array)
    
    # Add batch dimension
    face_array = np.expand_dims(face_array, axis=0)
    
    return face_array

def detect_emotion(face_image, model=None):
    """
    Detect emotion in a face image
    
    Args:
        face_image: RGB face image array
        model: Pre-loaded emotion model (optional)
        
    Returns:
        Predicted emotion label
    """
    # Load model if not provided
    if model is None:
        global _emotion_model
        if '_emotion_model' not in globals():
            _emotion_model = load_emotion_model()
        model = _emotion_model
    
    try:
        # Preprocess the face
        processed_face = preprocess_face_for_emotion(face_image)
        
        # Make prediction
        prediction = model.predict(processed_face, verbose=0)
        
        # Get the emotion with highest probability
        emotion_idx = np.argmax(prediction)
        emotion = EMOTIONS[emotion_idx]
        confidence = float(prediction[0][emotion_idx])
        
        return emotion, confidence
    
    except Exception as e:
        print(f"Error in emotion detection: {e}")
        return 'unknown', 0.0

def analyze_group_emotions(faces):
    """
    Analyze emotions of a group of faces
    
    Args:
        faces: List of face images
        
    Returns:
        emotions: List of emotion labels
        confidences: List of confidence scores
        dominant_emotion: Most common emotion in the group
    """
    if not faces:
        return [], [], None
    
    # Load model once
    model = load_emotion_model()
    
    emotions = []
    confidences = []
    
    for face in faces:
        emotion, confidence = detect_emotion(face, model)
        emotions.append(emotion)
        confidences.append(confidence)
    
    # Find dominant emotion (most common)
    if emotions:
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        dominant_emotion = max(emotion_counts, key=emotion_counts.get)
    else:
        dominant_emotion = None
    
    return emotions, confidences, dominant_emotion

def visualize_emotions(image, faces, bboxes, emotions):
    """
    Visualize detected emotions on faces
    
    Args:
        image: Original image
        faces: List of face images
        bboxes: List of face bounding boxes
        emotions: List of detected emotions
        
    Returns:
        Image with emotions visualized
    """
    # Make a copy to avoid modifying the original
    result = image.copy()
    
    # Color mapping for emotions
    emotion_colors = {
        'angry': (0, 0, 255),     # Red
        'disgust': (0, 128, 0),   # Dark Green
        'fear': (128, 0, 128),    # Purple
        'happy': (0, 255, 0),     # Green
        'sad': (255, 0, 0),       # Blue
        'surprise': (255, 255, 0), # Cyan
        'neutral': (128, 128, 128), # Gray
        'unknown': (200, 200, 200)  # Light Gray
    }
    
    # Draw bounding boxes and emotions
    for bbox, emotion in zip(bboxes, emotions):
        x1, y1, x2, y2 = bbox
        color = emotion_colors.get(emotion, (200, 200, 200))
        
        # Draw rectangle
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
        
        # Add emotion label
        cv2.putText(
            result, 
            emotion, 
            (x1, y1 - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.7, 
            color, 
            2
        )
    
    return result