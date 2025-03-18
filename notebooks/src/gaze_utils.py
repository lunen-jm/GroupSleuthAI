import cv2
import numpy as np
import dlib

def get_eyes_from_face(face, face_bbox):
    '''
    Extract eye regions from a face image using dlib facial landmarks
    
    Args:
        face: Face image
        face_bbox: Bounding box of face [x1, y1, x2, y2]
    
    Returns:
        left_eye: Left eye image region
        right_eye: Right eye image region
        left_eye_pts: Left eye landmark points
        right_eye_pts: Right eye landmark points
    '''
    # Initialize the dlib facial landmark detector
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")
    
    # Convert bbox to dlib rectangle
    x1, y1, x2, y2 = face_bbox
    dlib_rect = dlib.rectangle(x1, y1, x2, y2)
    
    # Get facial landmarks
    shape = predictor(cv2.cvtColor(face, cv2.COLOR_RGB2GRAY), dlib_rect)
    
    # Extract eye landmarks (36-41 for left eye, 42-47 for right eye)
    left_eye_pts = [(shape.part(i).x, shape.part(i).y) for i in range(36, 42)]
    right_eye_pts = [(shape.part(i).x, shape.part(i).y) for i in range(42, 48)]
    
    # Calculate eye bounding boxes
    left_eye_x = min(pt[0] for pt in left_eye_pts)
    left_eye_y = min(pt[1] for pt in left_eye_pts)
    left_eye_w = max(pt[0] for pt in left_eye_pts) - left_eye_x
    left_eye_h = max(pt[1] for pt in left_eye_pts) - left_eye_y
    
    right_eye_x = min(pt[0] for pt in right_eye_pts)
    right_eye_y = min(pt[1] for pt in right_eye_pts)
    right_eye_w = max(pt[0] for pt in right_eye_pts) - right_eye_x
    right_eye_h = max(pt[1] for pt in right_eye_pts) - right_eye_y
    
    # Add padding
    padding = 5
    left_eye_x = max(0, left_eye_x - padding)
    left_eye_y = max(0, left_eye_y - padding)
    left_eye_w += 2 * padding
    left_eye_h += 2 * padding
    
    right_eye_x = max(0, right_eye_x - padding)
    right_eye_y = max(0, right_eye_y - padding)
    right_eye_w += 2 * padding
    right_eye_h += 2 * padding
    
    # Extract eye regions
    left_eye = face[left_eye_y:left_eye_y+left_eye_h, left_eye_x:left_eye_x+left_eye_w]
    right_eye = face[right_eye_y:right_eye_y+right_eye_h, right_eye_x:right_eye_x+right_eye_w]
    
    # Adjust eye points to be relative to the eye regions
    left_eye_pts = [(x - left_eye_x, y - left_eye_y) for x, y in left_eye_pts]
    right_eye_pts = [(x - right_eye_x, y - right_eye_y) for x, y in right_eye_pts]
    
    return left_eye, right_eye, left_eye_pts, right_eye_pts

def estimate_gaze_direction(eye_region, eye_points):
    '''
    Estimate the gaze direction based on iris position within the eye
    
    Args:
        eye_region: Image of the eye region
        eye_points: Eye landmark points
    
    Returns:
        gaze_direction: Estimated gaze direction as string
        gaze_ratio: Numerical representation of gaze direction
        iris_center: Coordinates of detected iris center
    '''
    # Convert to grayscale
    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_RGB2GRAY)
    
    # Create mask for eye region
    mask = np.zeros_like(gray_eye)
    cv2.fillPoly(mask, [np.array(eye_points)], 255)
    
    # Apply mask and threshold to isolate iris
    masked_eye = cv2.bitwise_and(gray_eye, gray_eye, mask=mask)
    _, thresholded = cv2.threshold(masked_eye, 50, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours to identify iris
    contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Default values if iris detection fails
    iris_center = (eye_region.shape[1] // 2, eye_region.shape[0] // 2)
    gaze_ratio = 1.0
    
    if contours:
        # Get the largest contour (likely the iris)
        iris_contour = max(contours, key=cv2.contourArea)
        
        # Calculate centroid of iris
        M = cv2.moments(iris_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            iris_center = (cx, cy)
        
        # Calculate eye width from points
        eye_left = min(pt[0] for pt in eye_points)
        eye_right = max(pt[0] for pt in eye_points)
        eye_width = eye_right - eye_left
        
        if eye_width > 0:
            # Calculate position ratio (0 = far left, 1 = center, 2 = far right)
            gaze_ratio = 2.0 * (iris_center[0] - eye_left) / eye_width
    
    # Determine gaze direction
    if gaze_ratio < 0.7:
        gaze_direction = "looking left"
    elif gaze_ratio > 1.3:
        gaze_direction = "looking right"
    else:
        gaze_direction = "looking center"
    
    return gaze_direction, gaze_ratio, iris_center

def analyze_gaze(face, face_bbox):
    '''
    Analyze gaze direction for a single face
    
    Args:
        face: Face image
        face_bbox: Bounding box of face [x1, y1, x2, y2]
    
    Returns:
        gaze_result: Dictionary with gaze analysis results
    '''
    # Extract eye regions
    try:
        left_eye, right_eye, left_eye_pts, right_eye_pts = get_eyes_from_face(face, face_bbox)
    except Exception as e:
        print(f"Error extracting eye regions: {e}")
        return {
            'success': False,
            'direction': "unknown",
            'left_gaze_ratio': None,
            'right_gaze_ratio': None
        }
    
    # Check if eye extraction was successful
    if left_eye.size == 0 or right_eye.size == 0:
        return {
            'success': False,
            'direction': "unknown",
            'left_gaze_ratio': None,
            'right_gaze_ratio': None
        }
    
    # Estimate gaze direction for each eye
    left_direction, left_ratio, left_iris = estimate_gaze_direction(left_eye, left_eye_pts)
    right_direction, right_ratio, right_iris = estimate_gaze_direction(right_eye, right_eye_pts)
    
    # Average the gaze ratios to get overall direction
    avg_ratio = (left_ratio + right_ratio) / 2
    
    # Determine overall gaze direction
    if avg_ratio < 0.7:
        overall_direction = "looking left"
    elif avg_ratio > 1.3:
        overall_direction = "looking right"
    else:
        overall_direction = "looking forward"
    
    # Return results
    return {
        'success': True,
        'direction': overall_direction,
        'left_gaze_ratio': left_ratio,
        'right_gaze_ratio': right_ratio,
        'avg_gaze_ratio': avg_ratio
    }

def analyze_group_gaze(faces, bboxes, face_positions=None):
    '''
    Analyze gaze patterns for a group of faces, including who might be looking at whom
    
    Args:
        faces: List of face images
        bboxes: List of face bounding boxes
        face_positions: Optional list of face center positions in the original image
    
    Returns:
        group_gaze_results: List of dictionaries with gaze analysis and potential targets
    '''
    # Analyze individual gazes
    individual_results = []
    for i, (face, bbox) in enumerate(zip(faces, bboxes)):
        results = analyze_gaze(face, [0, 0, face.shape[1], face.shape[0]])
        results['face_id'] = i
        individual_results.append(results)
    
    # If face positions are provided, estimate who might be looking at whom
    if face_positions is not None:
        for i, result in enumerate(individual_results):
            if not result['success'] or result['direction'] == "looking forward":
                # Can't determine gaze target if looking forward or failed analysis
                result['gaze_target_id'] = -1
                continue
            
            # Get current face position
            current_x, current_y = face_positions[i]
            
            # Determine likely gaze target based on direction and position
            potential_targets = []
            
            for j, (target_x, target_y) in enumerate(face_positions):
                if i == j:  # Skip self
                    continue
                    
                # Calculate angle to target
                dx = target_x - current_x
                dy = target_y - current_y
                
                # Skip targets behind the face if looking forward
                if result['direction'] == "looking left" and dx > 0:
                    continue
                if result['direction'] == "looking right" and dx < 0:
                    continue
                    
                # Calculate distance to target
                distance = np.sqrt(dx**2 + dy**2)
                
                # Add potential target
                potential_targets.append((j, distance))
            
            # Sort by distance and get closest target
            if potential_targets:
                potential_targets.sort(key=lambda x: x[1])
                result['gaze_target_id'] = potential_targets[0][0]
            else:
                result['gaze_target_id'] = -1
    
    return individual_results
