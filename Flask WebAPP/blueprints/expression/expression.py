from flask import Flask, render_template, Response, Blueprint, request, jsonify
import cv2
import numpy as np
from keras import metrics
from keras.models import load_model
from keras.preprocessing.image import load_img
from tqdm import tqdm
import os
from werkzeug.utils import secure_filename
import base64
from io import BytesIO
from PIL import Image
import time
import threading
import uuid
import collections
from collections import deque
import statistics

# FIXED: Removed url_prefix from blueprint since it's set in app.py
expression_bp = Blueprint(
    "expression", 
    __name__, 
    static_folder='static', 
    template_folder='templates',
    static_url_path='/static'
)

# Configuration
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
MAX_FILE_SIZE = 16 * 1024 * 1024  # 16MB max file size

# Global variables for models (initialize once)
emotion_model = None
age_model = None
models_loaded = False
model_lock = threading.Lock()

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
gender_mapping = {0: 'Male', 1: 'Female'}

# Camera and latest predictions
camera = None
camera_lock = threading.Lock()
latest_emotion = "N/A"
latest_age = "N/A"

# Stability improvements - prediction history and frame skipping
prediction_history = {
    'age': deque(maxlen=10),  # Store last 10 age predictions
    'gender': deque(maxlen=10),  # Store last 10 gender predictions
    'emotion': deque(maxlen=5)  # Store last 5 emotion predictions
}

frame_skip_counter = 0
FRAME_SKIP_INTERVAL = 3  # Only predict every 3rd frame for stability

# Stable tracking variables
stable_age = "N/A"
stable_gender = "N/A"
stable_emotion = "N/A"

def init_models():
    """Initialize models with error handling and retries."""
    global emotion_model, age_model, models_loaded
    
    with model_lock:
        if models_loaded:
            return True
            
        try:
            print("Loading emotion model...")
            emotion_model = load_model("blueprints/expression/expression_emotion_model.h5")
            print("Emotion model loaded successfully")
            
            print("Loading age model...")
            age_model = load_model("blueprints/expression/age_model.h5", 
                                 custom_objects={'mae': metrics.MeanAbsoluteError()})
            print("Age model loaded successfully")
            
            models_loaded = True
            return True
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return False

def init_camera():
    """Initialize camera with error handling."""
    global camera
    
    with camera_lock:
        if camera is None:
            try:
                camera = cv2.VideoCapture(0)
                if not camera.isOpened():
                    print("Warning: Camera not available")
                    camera = None
                else:
                    print("Camera initialized successfully")
            except Exception as e:
                print(f"Error initializing camera: {str(e)}")
                camera = None

def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_image_file(file):
    """Validate uploaded image file."""
    if not file:
        return False, "No file provided"
    
    if file.filename == '':
        return False, "No file selected"
    
    if not allowed_file(file.filename):
        return False, "Invalid file type. Supported formats: JPG, PNG, GIF, BMP"
    
    # Check file size
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)
    
    if file_size > MAX_FILE_SIZE:
        return False, "File too large. Maximum size: 16MB"
    
    if file_size == 0:
        return False, "File is empty"
    
    return True, "File is valid"

def smooth_predictions(age_raw, gender_raw, emotion):
    """Apply temporal smoothing to predictions."""
    global prediction_history
    
    # Add current predictions to history
    prediction_history['age'].append(age_raw)
    prediction_history['gender'].append(gender_raw)
    prediction_history['emotion'].append(emotion)
    
    # Calculate smoothed values
    if len(prediction_history['age']) >= 3:
        # Use median for age to reduce outliers
        smoothed_age = statistics.median(prediction_history['age'])
        
        # Use mode for gender (most common prediction)
        gender_votes = list(prediction_history['gender'])
        if len(gender_votes) >= 3:
            try:
                smoothed_gender = statistics.mode(gender_votes)
            except statistics.StatisticsError:
                # If no mode, use most recent
                smoothed_gender = gender_votes[-1]
        else:
            smoothed_gender = gender_raw
            
        # Use mode for emotion
        emotion_votes = list(prediction_history['emotion'])
        if len(emotion_votes) >= 3:
            try:
                smoothed_emotion = statistics.mode(emotion_votes)
            except statistics.StatisticsError:
                # If no mode, use most recent
                smoothed_emotion = emotion_votes[-1]
        else:
            smoothed_emotion = emotion
            
        return smoothed_age, smoothed_gender, smoothed_emotion
    else:
        return age_raw, gender_raw, emotion

def preprocess_image_for_emotion(image):
    """Preprocess the image for emotion prediction."""
    try:
        # Ensure image is grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        image = cv2.GaussianBlur(image, (3, 3), 0)
        
        # Resize to 48x48 for emotion model
        image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_LANCZOS4)
        
        # Apply histogram equalization for better contrast
        image = cv2.equalizeHist(image)
        
        # Normalize pixel values to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Add channel dimension and batch dimension
        image = np.expand_dims(image, axis=-1)  # Add channel dimension
        image = np.expand_dims(image, axis=0)   # Add batch dimension
        
        return image
    except Exception as e:
        print(f"Error in preprocess_image_for_emotion: {str(e)}")
        return None

def preprocess_image_for_age_gender(image):
    """Improved preprocessing with better consistency."""
    try:
        # Ensure image is grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        image = cv2.GaussianBlur(image, (3, 3), 0)
        
        # Resize to 128x128 for age/gender model
        image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_LANCZOS4)
        
        # Apply gentle histogram equalization
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
        image = clahe.apply(image)
        
        # Normalize with consistent method
        image = image.astype(np.float32)
        image = (image - image.mean()) / (image.std() + 1e-7)  # Z-score normalization
        image = np.clip(image, -3, 3)  # Clip outliers
        image = (image + 3) / 6  # Normalize to [0, 1]
        
        # Add dimensions
        image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)
        
        return image
    except Exception as e:
        print(f"Error in preprocess_image_for_age_gender: {str(e)}")
        return None

def predict_emotion(face_image):
    """Predict emotion from face image with stability improvements."""
    try:
        if emotion_model is None:
            return "Model not loaded"
        
        processed_image = preprocess_image_for_emotion(face_image)
        if processed_image is None:
            return "Preprocessing failed"
        
        # Get multiple predictions for stability
        predictions_list = []
        for _ in range(2):  # Run 2 predictions and average
            pred = emotion_model.predict(processed_image, verbose=0)
            predictions_list.append(pred)
        
        # Average the predictions
        avg_prediction = np.mean(predictions_list, axis=0)
        predicted_class = np.argmax(avg_prediction)
        confidence = np.max(avg_prediction)
        
        # Add confidence threshold to avoid low-confidence predictions
        if confidence < 0.4:
            return "Neutral"
        
        emotion = emotion_labels[predicted_class]
        print(f"Emotion prediction: {emotion} (confidence: {confidence:.3f})")
        
        return emotion
    except Exception as e:
        print(f"Error in predict_emotion: {str(e)}")
        return "Error"

def predict_age_improved(face_image):
    """Improved age and gender prediction with stability fixes."""
    try:
        if age_model is None:
            return "Model not loaded"
        
        processed_image = preprocess_image_for_age_gender(face_image)
        if processed_image is None:
            return "Preprocessing failed"
        
        # Get multiple predictions for stability
        predictions_list = []
        for _ in range(3):  # Run 3 predictions and average
            pred = age_model.predict(processed_image, verbose=0)
            predictions_list.append(pred)
        
        # Average the predictions
        avg_predictions = []
        for i in range(len(predictions_list[0])):
            avg_pred = np.mean([pred[i] for pred in predictions_list], axis=0)
            avg_predictions.append(avg_pred)
        
        predictions = avg_predictions
        
        if len(predictions) == 2:
            gender_prediction, age_prediction = predictions
        else:
            print(f"Unexpected model output format: {len(predictions)} outputs")
            return "Model format error"
        
        # More robust age processing
        age_raw = float(age_prediction[0][0])
        
        # Handle different age scales more consistently
        if 0 <= age_raw <= 1:
            # Normalized age (0-1)
            age = int(age_raw * 80 + 10)  # Map to 10-90 range
        elif 1 < age_raw <= 10:
            # Decade format
            age = int(age_raw * 10)
        elif age_raw > 100:
            # Very large scale
            age = int(age_raw / 100) if age_raw > 1000 else int(age_raw / 10)
        else:
            age = int(age_raw)
        
        # Clamp to reasonable bounds
        age = max(5, min(85, age))
        
        # More robust gender processing
        gender_raw = float(gender_prediction[0][0])
        
        # Use probability threshold with hysteresis
        if gender_raw > 0.7:
            gender_class = 1
        elif gender_raw < 0.3:
            gender_class = 0
        else:
            # Use previous prediction if uncertain
            if len(prediction_history['gender']) > 0:
                gender_class = prediction_history['gender'][-1]
            else:
                gender_class = 1 if gender_raw >= 0.5 else 0
        
        gender = gender_mapping[gender_class]
        
        print(f"Age/Gender prediction: {gender}, {age} years (raw: gender={gender_raw:.3f}, age={age_raw:.3f})")
        
        return age, gender_class, gender
        
    except Exception as e:
        print(f"Error in predict_age_improved: {str(e)}")
        import traceback
        traceback.print_exc()
        return "Error"

def predict_age(face_image):
    """Legacy function for backward compatibility."""
    result = predict_age_improved(face_image)
    if isinstance(result, tuple) and len(result) == 3:
        age, gender_class, gender = result
        return f"Gender: {gender}, Age: {age}"
    else:
        return result

def filter_overlapping_faces(faces, overlap_threshold=0.2):
    """Remove overlapping face detections using Non-Maximum Suppression."""
    if len(faces) <= 1:
        return faces
    
    # Convert to list of [x, y, x2, y2] format for easier IoU calculation
    boxes = []
    for (x, y, w, h) in faces:
        boxes.append([x, y, x + w, y + h])
    
    boxes = np.array(boxes, dtype=np.float32)
    
    # Calculate areas
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    
    # Sort by area (largest first) instead of bottom-right coordinate
    indices = np.argsort(areas)[::-1]
    
    keep = []
    while len(indices) > 0:
        # Always keep the largest remaining face
        i = indices[0]
        keep.append(i)
        
        if len(indices) == 1:
            break
        
        # Find the largest coordinates for intersection rectangle
        xx1 = np.maximum(boxes[i, 0], boxes[indices[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[indices[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[indices[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[indices[1:], 3])
        
        # Compute width and height of intersection rectangle
        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        
        # Compute intersection over union
        intersection = w * h
        union = areas[i] + areas[indices[1:]] - intersection
        overlap = intersection / union
        
        # Delete all indices that have IoU greater than threshold
        indices = np.delete(indices, np.concatenate(([0], np.where(overlap > overlap_threshold)[0] + 1)))
    
    return [faces[i] for i in keep]

def validate_face_region(face_region, min_size=30):
    """Validate face region quality."""
    if face_region.shape[0] < min_size or face_region.shape[1] < min_size:
        return False
    
    # Check if region has enough variation
    std_dev = np.std(face_region)
    if std_dev < 15:
        print(f"Face rejected: low variation (std: {std_dev:.2f})")
        return False
    
    # Check aspect ratio
    aspect_ratio = face_region.shape[1] / face_region.shape[0]
    if aspect_ratio < 0.6 or aspect_ratio > 1.8:
        print(f"Face rejected: bad aspect ratio ({aspect_ratio:.2f})")
        return False
    
    # Check for minimum face-like features
    edges = cv2.Canny(face_region, 50, 150)
    edge_density = np.sum(edges > 0) / (face_region.shape[0] * face_region.shape[1])
    if edge_density < 0.02:
        print(f"Face rejected: insufficient edges ({edge_density:.3f})")
        return False
    
    return True

def process_image_from_memory(image_data):
    """Process image from memory with improved face detection and prediction."""
    try:
        # Convert image data to numpy array
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return None, "Could not decode image. Please check the file format."
        
        # Validate image dimensions
        if image.shape[0] < 50 or image.shape[1] < 50:
            return None, "Image too small. Minimum size: 50x50 pixels"
        
        original_image = image.copy()
        
        # Convert to grayscale for face detection
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply CLAHE for better face detection
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray_image = clahe.apply(gray_image)
        
        # Face detection with more conservative parameters
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Primary detection with stricter parameters
        faces = face_cascade.detectMultiScale(
            gray_image, 
            scaleFactor=1.2,  # Increased from 1.1
            minNeighbors=8,   # Increased from 5
            minSize=(60, 60), # Increased minimum size
            maxSize=(min(image.shape[0], image.shape[1])//2, min(image.shape[0], image.shape[1])//2),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        print(f"Initial detection found {len(faces)} faces")
        
        # If no faces found, try with slightly relaxed parameters (but still conservative)
        if len(faces) == 0:
            faces = face_cascade.detectMultiScale(
                gray_image, 
                scaleFactor=1.15,  # More conservative than before
                minNeighbors=6,    # Still higher than original
                minSize=(50, 50),  # Slightly smaller minimum
                maxSize=(min(image.shape[0], image.shape[1])//2, min(image.shape[0], image.shape[1])//2),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            print(f"Secondary detection found {len(faces)} faces")
        
        if len(faces) == 0:
            return None, "No faces detected in the image. Please ensure the image contains clear, well-lit faces."
        
        # More aggressive overlap filtering
        if len(faces) > 1:
            faces = filter_overlapping_faces(faces, overlap_threshold=0.2)  # Reduced from 0.3
            print(f"After filtering overlaps: {len(faces)} faces remain")
        
        # Additional size-based filtering - keep only largest faces
        if len(faces) > 1:
            # Sort by area (width * height) in descending order
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            # Keep only the largest face if multiple faces are very close in size
            largest_area = faces[0][2] * faces[0][3]
            filtered_faces = [faces[0]]  # Always keep the largest
            
            for face in faces[1:]:
                current_area = face[2] * face[3]
                # Only keep additional faces if they're at least 50% the size of the largest
                if current_area >= (largest_area * 0.5):
                    filtered_faces.append(face)
            
            faces = filtered_faces
            print(f"After size filtering: {len(faces)} faces remain")
        
        # Validate face regions with stricter criteria
        valid_faces = []
        for (x, y, w, h) in faces:
            face_region = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)[y:y + h, x:x + w]
            if validate_face_region(face_region, min_size=40):  # Increased min_size
                valid_faces.append((x, y, w, h))
        
        faces = valid_faces
        print(f"After validation: {len(faces)} valid faces")
        
        if len(faces) == 0:
            return None, "No valid faces found for processing."
        
        # Limit to maximum 3 faces (reduced from 5)
        if len(faces) > 3:
            faces = faces[:3]
            print(f"Limited to top 3 largest faces")
        
        # Rest of the function remains the same...
        results = []
        
        # Process each face
        for i, (x, y, w, h) in enumerate(faces):
            face_image_bgr = original_image[y:y + h, x:x + w]
            face_image_gray = cv2.cvtColor(face_image_bgr, cv2.COLOR_BGR2GRAY)
            
            if not validate_face_region(face_image_gray, min_size=40):
                print(f"Skipping face {i+1} - failed validation")
                continue
            
            # Apply consistent preprocessing
            face_image_enhanced = cv2.GaussianBlur(face_image_gray, (3, 3), 0)
            face_image_enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(face_image_enhanced)
            
            # Predict emotion and age/gender
            emotion = predict_emotion(face_image_enhanced)
            age_result = predict_age_improved(face_image_enhanced)
            
            # Handle age/gender result
            if isinstance(age_result, tuple) and len(age_result) == 3:
                age, gender_class, gender = age_result
                age_gender = f"Gender: {gender}, Age: {age}"
            else:
                age_gender = age_result
            
            # Skip if predictions failed
            if emotion in ["Error", "Model not loaded", "Preprocessing failed"] or age_gender in ["Error", "Model not loaded"]:
                print(f"Skipping face {i+1} - prediction failed")
                continue
            
            # Draw annotations
            cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add background rectangles for text
            cv2.rectangle(original_image, (x, y - 35), (x + w, y), (0, 0, 0), -1)
            cv2.rectangle(original_image, (x, y + h), (x + w, y + h + 35), (0, 0, 0), -1)
            
            # Add text labels
            cv2.putText(original_image, emotion, (x + 5, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(original_image, age_gender, (x + 5, y + h + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            results.append({
                'face_id': i + 1,
                'emotion': emotion,
                'age_gender': age_gender,
                'coordinates': {'x': int(x), 'y': int(y), 'w': int(w), 'h': int(h)}
            })
        
        if results:
            # Encode processed image to base64
            _, buffer = cv2.imencode('.jpg', original_image)
            processed_image_b64 = base64.b64encode(buffer).decode('utf-8')
            
            print(f"Successfully processed {len(results)} faces")
            return results, processed_image_b64
        else:
            return None, "No valid faces found for processing after filtering"
        
    except Exception as e:
        print(f"Error in process_image_from_memory: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, f"Error processing image: {str(e)}"

# Routes
@expression_bp.route('/')
def index():
    """Main page route."""
    if not models_loaded:
        init_models()
    init_camera()
    return render_template('expression_index.html')

@expression_bp.route('/video_feed')
def video_feed():
    """Video feed route for webcam with improved stability."""
    if not models_loaded:
        init_models()
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@expression_bp.route('/latest_emotion')
def get_latest_emotion():
    """Get latest emotion prediction."""
    return jsonify({'emotion': latest_emotion})

@expression_bp.route('/latest_age_gender')
def get_latest_age_gender():
    """Get latest age/gender prediction."""
    return jsonify({'age_gender': latest_age})

@expression_bp.route('/upload_image', methods=['POST'])
def upload_image():
    """Handle image upload and analysis."""
    try:
        print("Upload image route called")
        
        if not models_loaded:
            print("Models not loaded, initializing...")
            success = init_models()
            if not success:
                print("Failed to initialize models")
                return jsonify({'error': 'Models failed to load. Please try again.'}), 500
        
        if 'file' not in request.files:
            print("No file in request")
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        print(f"File received: {file.filename}")
        
        file_data = file.read()
        print(f"File size: {len(file_data)} bytes")
        
        file.seek(0)
        
        is_valid, message = validate_image_file(file)
        if not is_valid:
            print(f"File validation failed: {message}")
            return jsonify({'error': message}), 400
        
        print("Processing image...")
        results, processed_image_b64 = process_image_from_memory(file_data)
        
        if results is None:
            print(f"Image processing failed: {processed_image_b64}")
            return jsonify({'error': processed_image_b64}), 400
        
        print(f"Image processing successful. Found {len(results)} faces")
        
        return jsonify({
            'success': True,
            'results': results,
            'processed_image': processed_image_b64,
            'total_faces': len(results)
        })
        
    except Exception as e:
        print(f"Exception in upload_image: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@expression_bp.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'models_loaded': models_loaded,
        'camera_available': camera is not None,
        'prediction_history_size': {
            'age': len(prediction_history['age']),
            'gender': len(prediction_history['gender']),
            'emotion': len(prediction_history['emotion'])
        }
    })

@expression_bp.route('/reset_predictions')
def reset_predictions():
    """Reset prediction history for fresh start."""
    global prediction_history, stable_age, stable_gender, stable_emotion
    global latest_emotion, latest_age, frame_skip_counter
    
    # Clear prediction history
    prediction_history = {
        'age': deque(maxlen=10),
        'gender': deque(maxlen=10),
        'emotion': deque(maxlen=5)
    }
    
    # Reset stable values
    stable_age = "N/A"
    stable_gender = "N/A"
    stable_emotion = "N/A"
    
    # Reset latest values
    latest_emotion = "N/A"
    latest_age = "N/A"
    frame_skip_counter = 0
    
    return jsonify({'success': True, 'message': 'Predictions reset successfully'})

@expression_bp.route('/camera_control', methods=['POST'])
def camera_control():
    """Control camera on/off."""
    global camera
    
    try:
        action = request.json.get('action', '').lower()
        
        with camera_lock:
            if action == 'start':
                if camera is None:
                    init_camera()
                    if camera is not None:
                        return jsonify({'success': True, 'message': 'Camera started'})
                    else:
                        return jsonify({'success': False, 'message': 'Failed to start camera'})
                else:
                    return jsonify({'success': True, 'message': 'Camera already running'})
            
            elif action == 'stop':
                if camera is not None:
                    camera.release()
                    camera = None
                    return jsonify({'success': True, 'message': 'Camera stopped'})
                else:
                    return jsonify({'success': True, 'message': 'Camera already stopped'})
            
            else:
                return jsonify({'success': False, 'message': 'Invalid action. Use "start" or "stop"'})
                
    except Exception as e:
        print(f"Error in camera_control: {str(e)}")
        return jsonify({'success': False, 'message': f'Error: {str(e)}'})

@expression_bp.route('/model_info')
def model_info():
    """Get information about loaded models."""
    info = {
        'models_loaded': models_loaded,
        'emotion_model_loaded': emotion_model is not None,
        'age_model_loaded': age_model is not None,
        'emotion_labels': emotion_labels,
        'gender_mapping': gender_mapping,
        'frame_skip_interval': FRAME_SKIP_INTERVAL,
        'prediction_history_sizes': {
            'age': prediction_history['age'].maxlen,
            'gender': prediction_history['gender'].maxlen,
            'emotion': prediction_history['emotion'].maxlen
        }
    }
    return jsonify(info)

@expression_bp.route('/get_predictions')
def get_predictions():
    """Get current stable predictions."""
    return jsonify({
        'stable_emotion': stable_emotion,
        'stable_age': stable_age,
        'stable_gender': stable_gender,
        'prediction_confidence': {
            'age_samples': len(prediction_history['age']),
            'gender_samples': len(prediction_history['gender']),
            'emotion_samples': len(prediction_history['emotion'])
        }
    })

# Error handlers
@expression_bp.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404

@expression_bp.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500

@expression_bp.errorhandler(413)
def too_large(error):
    """Handle file too large errors."""
    return jsonify({'error': 'File too large. Maximum size: 16MB'}), 413

# Cleanup function
def cleanup_resources():
    """Clean up resources on shutdown."""
    global camera
    try:
        with camera_lock:
            if camera is not None:
                camera.release()
                camera = None
        print("Resources cleaned up successfully")
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")

# Initialize models on import
if __name__ == "__main__":
    init_models()
    print("Expression detection blueprint initialized")
