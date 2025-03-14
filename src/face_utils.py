import cv2
import numpy as np

def detect_faces(image_path, confidence_threshold=0.5):
    """
    Detect faces in an image using OpenCV's DNN face detector
    
    Args:
        image_path: Path to the input image
        confidence_threshold: Minimum confidence for face detection
        
    Returns:
        faces: List of face images
        bboxes: List of bounding boxes in format [x1, y1, x2, y2]
        original_image: The original image
    """
    # Load the DNN model for face detection
    model_file = "../models/face_detector/opencv_face_detector_uint8.pb"
    config_file = "../models/face_detector/opencv_face_detector.pbtxt"
    
    try:
        net = cv2.dnn.readNetFromTensorflow(model_file, config_file)
    except:
        print(f"Could not load face detection model from {model_file}")
        print("Please download the model files first or use a different detector")
        return [], [], None
    
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image from {image_path}")
        return [], [], None
    
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    
    # Create a blob and pass it through the model
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), [104, 117, 123], False, False)
    net.setInput(blob)
    detections = net.forward()
    
    # Process detections
    faces = []
    bboxes = []
    
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > confidence_threshold:
            x1 = max(0, int(detections[0, 0, i, 3] * w))
            y1 = max(0, int(detections[0, 0, i, 4] * h))
            x2 = min(w, int(detections[0, 0, i, 5] * w))
            y2 = min(h, int(detections[0, 0, i, 6] * h))
            
            # Ensure the face region is valid
            if x2 <= x1 or y2 <= y1:
                continue
                
            # Extract face
            face = image_rgb[y1:y2, x1:x2]
            if face.size == 0:
                continue
                
            bboxes.append([x1, y1, x2, y2])
            faces.append(face)
    
    return faces, bboxes, image_rgb

def download_face_detector():
    """
    Downloads the OpenCV DNN face detector model files if they don't exist
    """
    import os
    
    models_dir = "../models/face_detector"
    os.makedirs(models_dir, exist_ok=True)
    
    # URLs for the model files
    model_url = "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/opencv_face_detector_uint8.pb"
    config_url = "https://github.com/opencv/opencv/raw/master/samples/dnn/face_detector/opencv_face_detector.pbtxt"
    
    # Download the files if they don't exist
    model_file = os.path.join(models_dir, "opencv_face_detector_uint8.pb")
    config_file = os.path.join(models_dir, "opencv_face_detector.pbtxt")
    
    if not os.path.exists(model_file):
        print(f"Downloading face detector model to {model_file}")
        import urllib.request
        urllib.request.urlretrieve(model_url, model_file)
    
    if not os.path.exists(config_file):
        print(f"Downloading face detector config to {config_file}")
        import urllib.request
        urllib.request.urlretrieve(config_url, config_file)