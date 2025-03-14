import cv2
import numpy as np
import dlib

def get_eyes_from_face(face_image, face_rect=None):
    """
    Extract left and right eye images from a face image
    
    Args:
        face_image: RGB image containing a face
        face_rect: Face rectangle coordinates (optional)
        
    Returns:
        left_eye: Left eye image
        right_eye: Right eye image
        eye_centers: Centers of left and right eyes
    """
    # Convert to grayscale for better detection
    gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
    
    # If face_rect is not provided, detect face
    if face_rect is None:
        faces = face_detector(gray)
        if len(faces) == 0:
            print("No face detected")
            return None, None, None
        face_rect = faces[0]  # Use the first detected face
    else:
        # Convert to dlib rectangle format
        x1, y1, x2, y2 = face_rect
        face_rect = dlib.rectangle(x1, y1, x2, y2)
    
    # Get facial landmarks
    landmarks = landmark_predictor(gray, face_rect)
    
    # Extract eye landmarks
    # Left eye points (36-41 in dlib's 68-point model)
    left_eye_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
    
    # Right eye points (42-47 in dlib's 68-point model)
    right_eye_points = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])
    
    # Get eye centers
    left_eye_center = np.mean(left_eye_points, axis=0).astype(int)
    right_eye_center = np.mean(right_eye_points, axis=0).astype(int)
    
    # Get bounding boxes for eyes
    left_eye_x1 = np.min(left_eye_points[:, 0]) - 5
    left_eye_y1 = np.min(left_eye_points[:, 1]) - 5
    left_eye_x2 = np.max(left_eye_points[:, 0]) + 5
    left_eye_y2 = np.max(left_eye_points[:, 1]) + 5
    
    right_eye_x1 = np.min(right_eye_points[:, 0]) - 5
    right_eye_y1 = np.min(right_eye_points[:, 1]) - 5
    right_eye_x2 = np.max(right_eye_points[:, 0]) + 5
    right_eye_y2 = np.max(right_eye_points[:, 1]) + 5
    
    # Ensure coordinates are within image bounds
    h, w = face_image.shape[:2]
    left_eye_x1, left_eye_y1 = max(0, left_eye_x1), max(0, left_eye_y1)
    left_eye_x2, left_eye_y2 = min(w, left_eye_x2), min(h, left_eye_y2)
    right_eye_x1, right_eye_y1 = max(0, right_eye_x1), max(0, right_eye_y1)
    right_eye_x2, right_eye_y2 = min(w, right_eye_x2), min(h, right_eye_y2)
    
    # Extract eye images
    left_eye = face_image[left_eye_y1:left_eye_y2, left_eye_x1:left_eye_x2]
    right_eye = face_image[right_eye_y1:right_eye_y2, right_eye_x1:right_eye_x2]
    
    return left_eye, right_eye, [left_eye_center, right_eye_center]


def estimate_gaze_direction(eye_image):
    """
    Estimate horizontal gaze direction from an eye image
    
    Args:
        eye_image: Image of an eye
        
    Returns:
        gaze_x: Horizontal gaze direction (-1 to 1, where 0 is center)
    """
    if eye_image is None or eye_image.size == 0:
        return 0  # Default: looking straight
    
    # Convert to grayscale
    gray_eye = cv2.cvtColor(eye_image, cv2.COLOR_RGB2GRAY)
    
    # Apply blur and threshold to isolate the pupil
    gray_eye = cv2.GaussianBlur(gray_eye, (7, 7), 0)
    
    # Use adaptive thresholding to handle different lighting conditions
    _, threshold_eye = cv2.threshold(gray_eye, 45, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours in the thresholded image
    contours, _ = cv2.findContours(threshold_eye, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 0:
        # Find the largest contour (likely the pupil)
        pupil_contour = max(contours, key=cv2.contourArea)
        
        # Get the centroid of the pupil
        M = cv2.moments(pupil_contour)
        if M["m00"] != 0:
            pupil_x = int(M["m10"] / M["m00"])
            
            # Calculate relative position in the eye (normalize from -1 to 1)
            eye_width = eye_image.shape[1]
            eye_center = eye_width // 2
            
            # Normalize to [-1, 1] where 0 is center
            # Negative means looking left, positive means looking right
            gaze_x = (pupil_x - eye_center) / (eye_width / 2)
            
            # Limit to valid range
            gaze_x = max(-1, min(1, gaze_x))
            
            return gaze_x
    
    # Default: looking straight
    return 0


def analyze_gaze(face_image, face_rect=None):
    """
    Analyze gaze direction of a face
    
    Args:
        face_image: RGB image containing a face
        face_rect: Face rectangle coordinates (optional)
        
    Returns:
        gaze_vector: Simplified gaze direction vector [x, y]
        face_with_gaze: Annotated face image showing gaze direction
    """
    # Extract eyes
    left_eye, right_eye, eye_centers = get_eyes_from_face(face_image, face_rect)
    
    if left_eye is None or right_eye is None:
        return [0, 0], face_image  # Default to looking straight ahead
    
    # Get gaze direction for both eyes
    left_gaze = estimate_gaze_direction(left_eye)
    right_gaze = estimate_gaze_direction(right_eye)
    
    # Average the directions
    avg_gaze_x = (left_gaze + right_gaze) / 2
    
    # For simplicity, we'll just use horizontal direction
    # Real gaze would include vertical component too
    gaze_vector = [avg_gaze_x, 0]
    
    # Create annotated image
    face_with_gaze = face_image.copy()
    
    # Draw eyes
    left_center, right_center = eye_centers
    cv2.circle(face_with_gaze, tuple(left_center), 3, (0, 255, 0), -1)
    cv2.circle(face_with_gaze, tuple(right_center), 3, (0, 255, 0), -1)
    
    # Calculate face center (average of both eyes)
    face_center = ((left_center[0] + right_center[0]) // 2, 
                   (left_center[1] + right_center[1]) // 2)
    
    # Draw gaze direction
    gaze_length = 50
    gaze_end = (
        int(face_center[0] + gaze_vector[0] * gaze_length),
        int(face_center[1] + gaze_vector[1] * gaze_length)
    )
    
    cv2.arrowedLine(face_with_gaze, face_center, gaze_end, (0, 0, 255), 2)
    
    return gaze_vector, face_with_gaze


def analyze_group_gaze(faces, boxes, positions):
    """
    Analyze gaze interactions between multiple people
    
    Args:
        faces: List of face images
        boxes: List of face bounding boxes
        positions: Positions of faces in the original image
        
    Returns:
        interactions: Dictionary describing gaze interactions
    """
    interactions = []
    gaze_vectors = []
    
    # Get gaze for each face
    for i, face in enumerate(faces):
        gaze_vector, _ = analyze_gaze(face)
        gaze_vectors.append(gaze_vector)
    
    # Calculate center points of each face
    face_centers = []
    for box in boxes:
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        face_centers.append((center_x, center_y))
    
    # Check if anyone is looking at someone else
    for i, gaze_vector in enumerate(gaze_vectors):
        source_center = face_centers[i]
        
        # Skip if looking straight ahead
        if abs(gaze_vector[0]) < 0.2:
            continue
        
        # Check direction
        looking_right = gaze_vector[0] > 0
        
        # For each other face, check if it's in the gaze direction
        for j, target_center in enumerate(face_centers):
            if i == j:  # Skip self
                continue
                
            # Check if target is in the right direction
            if looking_right and target_center[0] > source_center[0]:
                # Target is to the right, and person is looking right
                interactions.append({
                    'source': i,
                    'target': j,
                    'confidence': abs(gaze_vector[0])
                })
            elif not looking_right and target_center[0] < source_center[0]:
                # Target is to the left, and person is looking left
                interactions.append({
                    'source': i,
                    'target': j,
                    'confidence': abs(gaze_vector[0])
                })
    
    return interactions, gaze_vectors
