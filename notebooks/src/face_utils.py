# filepath: src/face_utils.py
import cv2
import numpy as np

def detect_faces(image_path):
    """
    Detect faces in an image using OpenCV's Haar Cascade
    
    Args:
        image_path: Path to the input image
        
    Returns:
        faces: List of face images
        bboxes: List of bounding boxes [x, y, w, h]
        original_image: The original image with faces marked
    """
    # Load the face detector
    face_cascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image from {image_path}")
        return [], [], None
    
    # Convert to RGB for display and grayscale for detection
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces_rect = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    # Process detected faces
    faces = []
    bboxes = []
    
    for (x, y, w, h) in faces_rect:
        # Extract the face region
        face_roi = image_rgb[y:y+h, x:x+w]
        
        # Save information
        faces.append(face_roi)
        bboxes.append([x, y, x+w, y+h])  # Convert to [x1, y1, x2, y2] format
    
    return faces, bboxes, image_rgb
