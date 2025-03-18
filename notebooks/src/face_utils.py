import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import mediapipe as mp
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
except ImportError:
    FACE_RECOGNITION_AVAILABLE = False
    print("Warning: face_recognition package not found. Some detection methods will be unavailable.")

def detect_faces(image_path):
    """
    Original function to detect faces in an image
    
    Args:
        image_path: Path to the image file
        
    Returns:
        faces: List of detected face images
        bboxes: List of bounding boxes in (x1, y1, x2, y2) format
        image: Original image
    """
    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Convert to RGB (OpenCV loads images in BGR)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Load the face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces_rect = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    
    # Extract face images and bounding boxes
    faces = []
    bboxes = []
    
    for (x, y, w, h) in faces_rect:
        x1, y1, x2, y2 = x, y, x+w, y+h
        face_img = image_rgb[y1:y2, x1:x2]
        faces.append(face_img)
        bboxes.append((x1, y1, x2, y2))
    
    return faces, bboxes, image_rgb

def multi_method_face_detection(
    image_path, 
    enhance_image=True, 
    visualize_steps=False,
    mediapipe_confidence=0.3,
    haar_min_neighbors=5,
    nms_threshold=0.5,
    include_sources=True,
    haar_max_detections=10,  # New parameter to limit Haar detections
    filter_haar=True         # New parameter to enable additional Haar filtering
):
    """
    Enhanced face detection using multiple detection methods combined
    with adjustable confidence parameters
    
    Args:
        image_path: Path to the image
        enhance_image: Whether to enhance the image before detection
        visualize_steps: Whether to show intermediate results
        mediapipe_confidence: MediaPipe detection confidence threshold (0.0-1.0)
        haar_min_neighbors: Minimum neighbors for Haar Cascade (higher = stricter)
        nms_threshold: Non-maximum suppression threshold (higher allows more overlapping faces)
        include_sources: Whether to include detection sources in the output
        haar_max_detections: Maximum number of Haar cascade detections to keep
        filter_haar: Apply extra filtering to Haar detections to reduce false positives
        
    Returns:
        faces: List of cropped face images
        bboxes: List of bounding boxes in (x1, y1, x2, y2) format
        image: Original image with detected faces
        sources: List of detection sources (only if include_sources=True)
    """
    # Read image
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"Could not read image from {image_path}")
    
    # Make a copy of the image for processing
    image = original_image.copy()
    
    # Convert to RGB for processing
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    
    # Step 1: Image enhancement for better face detection
    if enhance_image:
        # Adjust brightness and contrast
        alpha = 1.2  # Contrast control
        beta = 10    # Brightness control
        enhanced_image = cv2.convertScaleAbs(image_rgb, alpha=alpha, beta=beta)
        
        # Apply mild denoising
        enhanced_image = cv2.fastNlMeansDenoisingColored(enhanced_image, None, 5, 5, 7, 21)
    else:
        enhanced_image = image_rgb.copy()
    
    if visualize_steps:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image_rgb)
        plt.title("Original")
        plt.subplot(1, 2, 2)
        plt.imshow(enhanced_image)
        plt.title("Enhanced")
        plt.tight_layout()
        plt.show()
    
    # Step 2: Combined face detection approach
    all_bboxes = []
    detection_sources = []  # Track which method detected each face
    
    # Method 1: MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    
    print("Running MediaPipe detection...")
    # Try both close and full range models
    for model_idx in [0, 1]:  # 0: close-range, 1: full-range
        with mp_face_detection.FaceDetection(
            model_selection=model_idx,
            min_detection_confidence=mediapipe_confidence) as detector:
            
            # Try on both original and enhanced images
            for img in [image_rgb, enhanced_image]:
                results = detector.process(img)
                if results.detections:
                    for detection in results.detections:
                        bbox = detection.location_data.relative_bounding_box
                        x1 = max(0, int(bbox.xmin * width))
                        y1 = max(0, int(bbox.ymin * height))
                        w = int(bbox.width * width)
                        h = int(bbox.height * height)
                        x2 = min(width, x1 + w)
                        y2 = min(height, y1 + h)
                        
                        # Add some padding to improve face extraction
                        padding = int(h * 0.1)  # 10% padding
                        y1 = max(0, y1 - padding)
                        y2 = min(height, y2 + padding)
                        
                        all_bboxes.append((x1, y1, x2, y2))
                        detection_sources.append(f"MediaPipe-{model_idx}")
    
    # Method 2: OpenCV's Haar Cascade
    print("Running Haar Cascade detection...")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Store Haar detections separately for filtering
    haar_bboxes = []
    haar_sources = []
    
    # Try with different scaling factors
    for scale_factor in [1.1, 1.2]:
        for img in [cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY), 
                   cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2GRAY)]:
            faces = face_cascade.detectMultiScale(
                img, 
                scaleFactor=scale_factor, 
                minNeighbors=haar_min_neighbors)
            
            for (x, y, w, h) in faces:
                x1, y1, x2, y2 = x, y, x + w, y + h
                haar_bboxes.append((x1, y1, x2, y2))
                haar_sources.append(f"Haar-{scale_factor}")
    
    # Apply additional filtering to Haar detections if enabled
    if filter_haar and haar_bboxes:
        # Sort by size (larger faces are more likely to be real)
        haar_areas = [(x2-x1)*(y2-y1) for x1,y1,x2,y2 in haar_bboxes]
        sorted_idxs = np.argsort(haar_areas)[::-1]  # Largest first
        
        # Validate aspect ratio and size
        filtered_haar_idxs = []
        for idx in sorted_idxs:
            x1, y1, x2, y2 = haar_bboxes[idx]
            w, h = x2-x1, y2-y1
            aspect_ratio = w / h if h > 0 else 0
            
            # Valid face aspect ratio is typically 0.8-1.2
            # Valid minimum face size (adjust as needed)
            min_face_size = min(height, width) * 0.05  # 5% of image dimension
            
            if (0.65 < aspect_ratio < 1.35) and (w > min_face_size) and (h > min_face_size):
                filtered_haar_idxs.append(idx)
                
                # Break if we reach our limit
                if len(filtered_haar_idxs) >= haar_max_detections:
                    break
        
        # Keep only the filtered detections
        haar_bboxes = [haar_bboxes[i] for i in filtered_haar_idxs]
        haar_sources = [haar_sources[i] for i in filtered_haar_idxs]
    
        if visualize_steps:
            print(f"Filtered Haar detections from {len(sorted_idxs)} to {len(filtered_haar_idxs)}")
    
    # Add filtered Haar detections to the main list
    all_bboxes.extend(haar_bboxes[:haar_max_detections])
    detection_sources.extend(haar_sources[:haar_max_detections])
    
    # Method 3: Face Recognition library (uses dlib)
    if FACE_RECOGNITION_AVAILABLE:
        print("Running face_recognition detection...")
        try:
            # Try both on original and enhanced images
            for img in [image_rgb, enhanced_image]:
                face_locations = face_recognition.face_locations(img, model="hog")
                for (top, right, bottom, left) in face_locations:
                    all_bboxes.append((left, top, right, bottom))
                    detection_sources.append("face_recognition")
        except Exception as e:
            print(f"Face Recognition method failed: {e}")
    else:
        print("Skipping face_recognition detection (package not available)")
    
    # Count detections by method before NMS
    if visualize_steps:
        method_counts = {}
        for method in detection_sources:
            method_counts[method] = method_counts.get(method, 0) + 1
        
        print("\nDetections by method (before NMS):")
        for method, count in method_counts.items():
            print(f"  {method}: {count} faces")
        print(f"  Total: {len(all_bboxes)} faces\n")
    
    # Remove duplicates through non-maximum suppression
    final_bboxes = []
    final_sources = []
    
    if all_bboxes:
        # Convert to numpy array for NMS
        bboxes_array = np.array(all_bboxes)
        
        # Calculate areas
        x1 = bboxes_array[:, 0]
        y1 = bboxes_array[:, 1]
        x2 = bboxes_array[:, 2]
        y2 = bboxes_array[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        
        # Sort by area
        sorted_indices = np.argsort(areas)[::-1]
        
        # Apply NMS
        keep_indices = []
        while len(sorted_indices) > 0:
            i = sorted_indices[0]
            keep_indices.append(i)
            
            # Calculate IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[sorted_indices[1:]])
            yy1 = np.maximum(y1[i], y1[sorted_indices[1:]])
            xx2 = np.minimum(x2[i], x2[sorted_indices[1:]])
            yy2 = np.minimum(y2[i], y2[sorted_indices[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            overlap = (w * h) / areas[sorted_indices[1:]]
            
            # Remove overlapping boxes based on threshold
            inds = np.where(overlap <= nms_threshold)[0]
            sorted_indices = sorted_indices[inds + 1]
        
        # Final bounding boxes after NMS
        final_bboxes = [all_bboxes[i] for i in keep_indices]
        final_sources = [detection_sources[i] for i in keep_indices]
    
    print(f"Detected {len(final_bboxes)} faces after NMS filtering")
    
    # Count detections by method after NMS
    if visualize_steps and final_sources:
        method_counts = {}
        for method in final_sources:
            method_counts[method] = method_counts.get(method, 0) + 1
        
        print("\nDetections by method (after NMS):")
        for method, count in method_counts.items():
            print(f"  {method}: {count} faces")
    
    # Extract face images
    faces = []
    for (x1, y1, x2, y2) in final_bboxes:
        face = image_rgb[y1:y2, x1:x2]
        if face.size > 0:
            faces.append(face)
    
    # Return with or without sources based on parameter
    if include_sources:
        return faces, final_bboxes, image_rgb, final_sources
    else:
        return faces, final_bboxes, image_rgb

def detect_faces2(image_path, max_faces=0, filter_haar=True, haar_max_detections=10, media = .4, haar = 4, nms = .55):
    """
    Enhanced face detection using multiple methods with simplified parameters
    
    Args:
        image_path: Path to the image file
        confidence_level: Overall confidence level (0.0-1.0) 
                         Low values (0.1-0.3): More faces detected, some might be false positives
                         Medium values (0.4-0.6): Balanced detection
                         High values (0.7-0.9): Fewer but more certain faces
        max_faces: Maximum number of faces to return (0 = no limit)
        filter_haar: Apply extra filtering to reduce Haar cascade false positives
        haar_max_detections: Maximum number of Haar cascade detections to keep
                         
    Returns:
        faces: List of detected face images
        bboxes: List of bounding boxes in (x1, y1, x2, y2) format
        image: Original image
    """
    # Map the overall confidence level to method-specific parameters
    # Each method has different sensitivity to its parameters
    
    # MediaPipe confidence (0.1-0.7 is the useful range)
    #mediapipe_conf = 0.1 + (confidence_level * 0.6)
    
    # Haar cascade min neighbors (3-8 is the useful range)
    # Lower confidence = fewer neighbors = more detections
    #haar_min_neighbors = max(3, int(3 + (confidence_level * 5)))
    
    # NMS threshold (0.3-0.7 is the useful range)
    # Higher confidence = lower NMS threshold = fewer overlapping detections
    #nms_threshold = max(0.3, 0.7 - (confidence_level * 0.4))
    
    # Print the derived parameters if needed
    print(f"Using detection parameters:")
    print(f"  - MediaPipe confidence: {media:.2f}")
    print(f"  - Haar min neighbors: {haar}")
    print(f"  - NMS threshold: {nms:.2f}")
    print(f"  - Filter Haar: {filter_haar}")
    print(f"  - Max Haar detections: {haar_max_detections}")
    
    faces, bboxes, image = multi_method_face_detection(
        image_path,
        enhance_image=True,
        visualize_steps=False,
        mediapipe_confidence=media, # Adjust this value (0.1-0.9)
        haar_min_neighbors=haar, # Adjust this value (2-8)
        nms_threshold=nms, # Adjust this value (0.3-0.7)
        include_sources=False,  # Don't include sources to match original format
        filter_haar=filter_haar,
        haar_max_detections=haar_max_detections
    )
    
    # Limit the number of faces if specified
    if max_faces > 0 and max_faces < len(faces):
        print(f"Limiting to {max_faces} of {len(faces)} detected faces")
        faces = faces[:max_faces]
        bboxes = bboxes[:max_faces]
    
    return faces, bboxes, image

def visualize_face_detection(
    image_path, 
    #  # Overall confidence level (replacing individual params)
    max_faces=25,           # Maximum number of faces to display in the grid
    use_custom_params=False,  # Flag to use custom parameters instead of confidence_level
    mediapipe_confidence=0.4, # Custom parameter if needed
    haar_min_neighbors=5,     # Custom parameter if needed
    nms_threshold=0.55,        # Custom parameter if needed
    filter_haar=True,         # Filter Haar detections
    haar_max_detections=10    # Max Haar detections to allow
):
    """
    Visualize face detection with confidence level or specific parameters
    
    Args:
        image_path: Path to the image
        confidence_level: Overall confidence level (0.0-1.0) that controls all parameters
        max_faces: Maximum number of faces to display in the grid (default: 25)
        use_custom_params: Whether to use individual parameters instead of confidence_level
        mediapipe_confidence: Custom confidence threshold for MediaPipe (if use_custom_params=True)
        haar_min_neighbors: Custom min neighbors for Haar (if use_custom_params=True)
        nms_threshold: Custom threshold for non-max suppression (if use_custom_params=True)
        filter_haar: Apply extra filtering to reduce Haar cascade false positives
        haar_max_detections: Maximum number of Haar cascade detections to keep
    """
    # Map the overall confidence level to method-specific parameters if not using custom params
    #if not use_custom_params:
        # MediaPipe confidence (0.1-0.7 is the useful range)
    #    mediapipe_confidence = 0.1 + (confidence_level * 0.6)
        
        # Haar cascade min neighbors (3-8 is the useful range)
    #    haar_min_neighbors = max(3, int(3 + (confidence_level * 5)))
        
        # NMS threshold (0.3-0.7 is the useful range)
    #    nms_threshold = max(0.3, 0.7 - (confidence_level * 0.4))
        
        # Print the derived parameters
    #    print(f"Using confidence level {confidence_level} with derived parameters:")
    #    print(f"  - MediaPipe confidence: {mediapipe_confidence:.2f}")
    #    print(f"  - Haar min neighbors: {haar_min_neighbors}")
    #    print(f"  - NMS threshold: {nms_threshold:.2f}")
    #    print(f"  - Filter Haar: {filter_haar}")
    #    print(f"  - Max Haar detections: {haar_max_detections}")
    
    # Run detection with selected parameters
    faces, bboxes, image, sources = multi_method_face_detection(
        image_path, 
        mediapipe_confidence=mediapipe_confidence,
        haar_min_neighbors=haar_min_neighbors,
        nms_threshold=nms_threshold,
        visualize_steps=True,
        include_sources=True,  # Make sure we get the sources
        filter_haar=filter_haar,
        haar_max_detections=haar_max_detections
    )
    
    # Display detected faces
    plt.figure(figsize=(12, 10))
    plt.imshow(image)
    
    # Draw bounding boxes with different colors based on detection method
    colors = {
        'MediaPipe-0': 'lime',
        'MediaPipe-1': 'green',
        'Haar-1.1': 'blue',
        'Haar-1.2': 'cyan',
        'face_recognition': 'magenta'
    }
    
    # Limit the number of faces to display if specified
    display_count = len(bboxes)
    if max_faces > 0 and max_faces < len(bboxes):
        display_count = max_faces
        print(f"Limiting display to {max_faces} of {len(bboxes)} detected faces")
    
    # Draw bounding boxes for all faces (even if we limit the grid view)
    for i, ((x1, y1, x2, y2), source) in enumerate(zip(bboxes, sources)):
        color = colors.get(source, 'yellow')
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                         fill=False, edgecolor=color, linewidth=2)
        plt.gca().add_patch(rect)
        plt.text(x1, y1-10, f"Face {i+1}", color='white', fontsize=10,
               bbox=dict(facecolor=color, alpha=0.7))
    
    # Set title based on whether we're using confidence level or custom parameters
    
    plt.title(f"Face Detection (MediaPipe={mediapipe_confidence}, Haar={haar_min_neighbors}, NMS={nms_threshold})\n"
                 f"Found {len(faces)} faces", fontsize=14)
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Show grid of detected faces (limited by max_faces)
    if faces and max_faces > 0:  # Only show grid if max_faces > 0
        # Limit to the specified maximum number of faces
        faces_to_show = faces[:max_faces]
        sources_to_show = sources[:max_faces]
        
        # Calculate grid dimensions
        rows = min(5, int(np.ceil(len(faces_to_show) / 5)))
        cols = min(5, len(faces_to_show))
        
        plt.figure(figsize=(15, 3*rows))
        for i, (face, source) in enumerate(zip(faces_to_show, sources_to_show)):
            plt.subplot(rows, cols, i+1)
            plt.imshow(face)
            plt.title(f"Face {i+1}: {source}")
            plt.axis('off')
        
        plt.suptitle(f"Extracted Faces (showing {len(faces_to_show)} of {len(faces)})", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95)
        plt.show()

def draw_faces_on_image(image, bboxes, labels=None):
    """
    Draw face bounding boxes on an image
    
    Args:
        image: Image to draw on
        bboxes: List of bounding boxes in (x1, y1, x2, y2) format
        labels: Optional list of labels for each box
        
    Returns:
        Image with faces drawn
    """
    # Make a copy of the image to avoid modifying the original
    img_copy = image.copy()
    
    # Convert from RGB to BGR if needed (for OpenCV)
    if len(img_copy.shape) == 3 and img_copy.shape[2] == 3:
        if img_copy.dtype != np.uint8:
            img_copy = (img_copy * 255).astype(np.uint8)
    
    # Draw each bounding box
    for i, (x1, y1, x2, y2) in enumerate(bboxes):
        # Draw rectangle
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Add label if provided
        if labels and i < len(labels):
            label = str(labels[i])
            # Position label above the box
            cv2.putText(img_copy, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return img_copy

# Helper function to calculate IoU (Intersection over Union)
def calculate_iou(box1, box2):
    """Calculate IoU between two bounding boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate intersection
    x_left = max(x1_1, x1_2)
    y_top = max(y1_1, y1_2)
    x_right = min(x2_1, x2_2)
    y_bottom = min(y2_1, y2_2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    
    # Calculate box areas
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    
    # Calculate IoU
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return iou