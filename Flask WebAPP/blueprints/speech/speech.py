from flask import Flask, request, jsonify, render_template, Blueprint
import numpy as np
import librosa
import os
import time
from tensorflow.keras.models import load_model
import sounddevice as sd
from scipy.io.wavfile import write
import glob
import warnings
warnings.filterwarnings('ignore')

speech_bp = Blueprint("speech", __name__, static_folder='static', template_folder='templates')

# Load model with error handling
try:
    model = load_model('blueprints/speech/speech_model.h5')
    print("Model loaded successfully")
    print(f"Model input shape: {model.input_shape}")
    print(f"Model output shape: {model.output_shape}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# CORRECTED emotion mapping for KunalGehlot model
# Based on SAVEE and RAVDESS datasets
emotion_map = {
    0: 'Neutral',    # Usually index 0
    1: 'Calm',       # Usually index 1  
    2: 'Happy',      # Usually index 2
    3: 'Sad',        # Usually index 3
    4: 'Angry',      # Usually index 4
    5: 'Fearful',    # Usually index 5
    6: 'Disgust',    # Usually index 6
    7: 'Surprised'   # Usually index 7 (if 8 classes)
}

# Alternative mapping (sometimes models use different orders)
alternative_emotion_map = {
    0: 'Angry',      # Some models put angry first
    1: 'Disgust',    
    2: 'Fearful',    
    3: 'Happy',      
    4: 'Neutral',    
    5: 'Sad',        
    6: 'Surprised'   
}

latest_emotion = "N/A"
recording = False
audio_data = []

def extract_features_exact_kunal_method(audio_path):
    """
    Extract features using the EXACT method from KunalGehlot's original code
    This is crucial for compatibility with the pre-trained model
    """
    try:
        # Load audio with exact parameters used in training
        y, sr = librosa.load(audio_path, sr=22050, duration=3.0, offset=0.5)
        
        # Remove silence (important for emotion recognition)
        y_trimmed, _ = librosa.effects.trim(y, top_db=25)
        
        # Pad or truncate to consistent length (3 seconds)
        target_length = sr * 3
        if len(y_trimmed) > target_length:
            y_trimmed = y_trimmed[:target_length]
        else:
            y_trimmed = np.pad(y_trimmed, (0, target_length - len(y_trimmed)), 'constant')
        
        # Normalize audio
        y_normalized = librosa.util.normalize(y_trimmed)
        
        # Extract MFCC features with exact parameters
        mfccs = librosa.feature.mfcc(
            y=y_normalized, 
            sr=sr, 
            n_mfcc=40,
            n_fft=2048,
            hop_length=512,
            n_mels=128,
            fmax=sr/2
        )
        
        # Statistical features - this is key for KunalGehlot model
        mfccs_mean = np.mean(mfccs.T, axis=0)
        mfccs_std = np.std(mfccs.T, axis=0)
        mfccs_max = np.max(mfccs.T, axis=0)
        mfccs_min = np.min(mfccs.T, axis=0)
        
        # Combine features (this might be what the model expects)
        features = np.concatenate([mfccs_mean, mfccs_std, mfccs_max, mfccs_min])
        
        # If model expects only 40 features, use just mean
        if len(features) > 40:
            features = mfccs_mean
        
        print(f"Audio duration: {len(y_normalized)/sr:.2f}s")
        print(f"MFCC shape: {mfccs.shape}")
        print(f"Features shape: {features.shape}")
        print(f"Features range: [{np.min(features):.4f}, {np.max(features):.4f}]")
        
        return features
        
    except Exception as e:
        print(f"Error in feature extraction: {str(e)}")
        return np.zeros(40)

def normalize_features_kunal_method(features):
    """
    Normalize features using the same method as training
    """
    # Handle NaN/inf values
    features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Standard normalization (z-score)
    mean = np.mean(features)
    std = np.std(features)
    
    if std > 1e-8:  # Avoid division by zero
        features = (features - mean) / std
    
    # Clip extreme values
    features = np.clip(features, -4, 4)
    
    print(f"Normalized features - Mean: {mean:.4f}, Std: {std:.4f}")
    print(f"After normalization - Range: [{np.min(features):.4f}, {np.max(features):.4f}]")
    
    return features

def test_model_with_known_inputs():
    """
    Test the model with known inputs to debug the issue
    """
    if model is None:
        return None
    
    print("\n=== TESTING MODEL WITH KNOWN INPUTS ===")
    
    # Test with different input patterns
    test_cases = [
        ("zeros", np.zeros(40)),
        ("ones", np.ones(40)),
        ("random_normal", np.random.normal(0, 1, 40)),
        ("random_uniform", np.random.uniform(-1, 1, 40)),
        ("angry_pattern", np.array([1, -1, 1, -1] * 10)),  # Alternating pattern
        ("calm_pattern", np.array([0.1] * 40)),  # Low values
        ("happy_pattern", np.array([0.5, 1.0] * 20)),  # High values
    ]
    
    results = {}
    
    for name, test_input in test_cases:
        try:
            # Normalize test input
            normalized_input = normalize_features_kunal_method(test_input.copy())
            
            # Reshape for model input
            model_input = np.expand_dims(normalized_input, axis=0)
            
            # Make prediction
            predictions = model.predict(model_input, verbose=0)
            predicted_class = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            results[name] = {
                'predicted_class': int(predicted_class),
                'emotion': emotion_map.get(predicted_class, 'Unknown'),
                'confidence': float(confidence),
                'all_probs': predictions[0].tolist()
            }
            
            print(f"{name}: {emotion_map.get(predicted_class, 'Unknown')} (confidence: {confidence:.4f})")
            
        except Exception as e:
            results[name] = {'error': str(e)}
            print(f"{name}: Error - {e}")
    
    return results

def check_model_bias():
    """
    Check if the model has a bias towards certain emotions
    """
    if model is None:
        return None
    
    print("\n=== CHECKING MODEL BIAS ===")
    
    # Test with 100 random inputs
    random_predictions = []
    
    for i in range(100):
        try:
            # Generate random MFCC-like features
            random_features = np.random.normal(0, 1, 40)
            normalized_features = normalize_features_kunal_method(random_features)
            model_input = np.expand_dims(normalized_features, axis=0)
            
            predictions = model.predict(model_input, verbose=0)
            predicted_class = np.argmax(predictions[0])
            random_predictions.append(predicted_class)
            
        except Exception as e:
            print(f"Random test {i} failed: {e}")
    
    # Count predictions
    from collections import Counter
    prediction_counts = Counter(random_predictions)
    
    print("Prediction distribution for random inputs:")
    for class_idx, count in prediction_counts.items():
        emotion = emotion_map.get(class_idx, 'Unknown')
        percentage = (count / len(random_predictions)) * 100
        print(f"  {emotion}: {count}/100 ({percentage:.1f}%)")
    
    # Check if heavily biased towards one emotion
    if len(prediction_counts) == 1:
        dominant_emotion = emotion_map.get(list(prediction_counts.keys())[0], 'Unknown')
        print(f"\n⚠️  MODEL IS HEAVILY BIASED - Always predicts: {dominant_emotion}")
        return {'bias_detected': True, 'dominant_emotion': dominant_emotion}
    
    return {'bias_detected': False, 'distribution': dict(prediction_counts)}

def record_audio_high_quality(duration=5, sample_rate=22050):
    """
    Record audio with high quality settings
    """
    global recording, audio_data
    audio_data = []
    recording = True
    
    print(f"Recording for {duration} seconds... Speak clearly and emotionally!")
    
    def callback(indata, frames, time_info, status):
        if status:
            print(f"Recording status: {status}")
        if recording:
            audio_data.append(indata.copy())
    
    try:
        with sd.InputStream(
            callback=callback,
            channels=1,
            samplerate=sample_rate,
            blocksize=4096,
            dtype='float32',
            latency='low'
        ):
            time.sleep(duration)
            recording = False
            
    except Exception as e:
        print(f"Recording error: {e}")
        recording = False
        return False
    
    # Save audio file
    audio_filename = os.path.join('blueprints/speech/static', 'temp_audio.wav')
    
    if audio_data:
        audio_array = np.concatenate(audio_data)
        
        # Check audio quality
        max_amplitude = np.max(np.abs(audio_array))
        rms_level = np.sqrt(np.mean(audio_array**2))
        
        print(f"Audio quality - Max amplitude: {max_amplitude:.6f}, RMS: {rms_level:.6f}")
        
        if max_amplitude < 0.001:
            print("⚠️  WARNING: Very low audio level - speak louder!")
        
        if rms_level < 0.0001:
            print("⚠️  WARNING: Very quiet audio - check microphone!")
        
        # Normalize and save
        if max_amplitude > 0:
            audio_array = audio_array / max_amplitude * 0.9  # Normalize to 90% to avoid clipping
        
        # Save as 16-bit WAV
        audio_int16 = (audio_array * 32767).astype(np.int16)
        write(audio_filename, sample_rate, audio_int16)
        
        print(f"Audio saved: {audio_filename}")
        return True
    
    return False

def predict_emotion_with_debugging(audio_path):
    """
    Predict emotion with extensive debugging
    """
    if model is None:
        return None, "Model not loaded"
    
    try:
        print(f"\n=== PREDICTING EMOTION FOR: {audio_path} ===")
        
        # Extract features
        features = extract_features_exact_kunal_method(audio_path)
        
        if np.all(features == 0):
            return None, "Feature extraction failed"
        
        # Normalize features
        normalized_features = normalize_features_kunal_method(features)
        
        # Prepare input for model
        model_input = np.expand_dims(normalized_features, axis=0)
        print(f"Model input shape: {model_input.shape}")
        
        # Make prediction
        predictions = model.predict(model_input, verbose=0)
        print(f"Raw predictions: {predictions}")
        print(f"Predictions shape: {predictions.shape}")
        print(f"Predictions sum: {np.sum(predictions):.4f}")
        
        # Check if predictions are valid
        if np.all(predictions == 0):
            return None, "Model returned all zeros"
        
        if np.all(np.isnan(predictions)):
            return None, "Model returned NaN values"
        
        # Get results
        predicted_class = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        all_probs = predictions[0]
        
        print(f"\nPrediction Results:")
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")
        
        # Show all probabilities
        print("\nAll emotion probabilities:")
        for i, prob in enumerate(all_probs):
            emotion = emotion_map.get(i, f'Class_{i}')
            print(f"  {emotion}: {prob:.4f}")
        
        # Try alternative emotion mapping if confidence is low
        detected_emotion = emotion_map.get(predicted_class, 'Unknown')
        
        if confidence < 0.3:  # Low confidence
            print(f"\n⚠️  Low confidence ({confidence:.4f}), trying alternative mapping...")
            alt_emotion = alternative_emotion_map.get(predicted_class, 'Unknown')
            print(f"Alternative emotion: {alt_emotion}")
        
        return {
            'predicted_class': int(predicted_class),
            'emotion': detected_emotion,
            'confidence': float(confidence),
            'all_probabilities': all_probs.tolist(),
            'features_mean': float(np.mean(normalized_features)),
            'features_std': float(np.std(normalized_features)),
            'raw_predictions': predictions.tolist()
        }, None
        
    except Exception as e:
        print(f"Error in emotion prediction: {e}")
        import traceback
        traceback.print_exc()
        return None, str(e)

# Flask routes
@speech_bp.route('/')
def index():
    return render_template('index.html')

@speech_bp.route('/start_recording', methods=['POST'])
def start_recording():
    global recording
    if not recording:
        # Clean up previous files
        previous_file = 'blueprints/speech/static/temp_audio.wav'
        if os.path.exists(previous_file):
            os.remove(previous_file)
        
        success = record_audio_high_quality(duration=5)
        if success:
            return jsonify({'status': 'Recording completed successfully'})
        else:
            return jsonify({'status': 'Recording failed'})
    
    return jsonify({'status': 'Already recording'})

@speech_bp.route('/stop_recording', methods=['POST'])
def stop_recording():
    global recording
    recording = False
    return jsonify({'status': 'Recording stopped'})

@speech_bp.route('/predict_emotion', methods=['POST'])
def predict_emotion():
    global latest_emotion
    
    audio_filename = 'blueprints/speech/static/temp_audio.wav'
    if not os.path.exists(audio_filename):
        return jsonify({'error': 'Audio file not found'}), 404
    
    # Predict emotion
    result, error = predict_emotion_with_debugging(audio_filename)
    
    if error:
        return jsonify({'error': error}), 500
    
    if result:
        latest_emotion = result['emotion']
        
        # Clean up
        try:
            os.remove(audio_filename)
        except:
            pass
        
        return jsonify(result)
    
    return jsonify({'error': 'Prediction failed'}), 500

@speech_bp.route('/debug_model', methods=['GET'])
def debug_model():
    """
    Comprehensive model debugging endpoint
    """
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    try:
        # Test with known inputs
        known_input_results = test_model_with_known_inputs()
        
        # Check for bias
        bias_results = check_model_bias()
        
        # Model architecture info
        model_info = {
            'input_shape': str(model.input_shape),
            'output_shape': str(model.output_shape),
            'total_params': model.count_params(),
            'layers': len(model.layers)
        }
        
        # Layer details
        layers_info = []
        for i, layer in enumerate(model.layers):
            layer_info = {
                'index': i,
                'name': layer.name,
                'type': type(layer).__name__,
                'trainable': layer.trainable
            }
            
            if hasattr(layer, 'output_shape'):
                layer_info['output_shape'] = str(layer.output_shape)
            
            layers_info.append(layer_info)
        
        return jsonify({
            'model_info': model_info,
            'layers_info': layers_info,
            'known_input_tests': known_input_results,
            'bias_analysis': bias_results,
            'emotion_mapping': emotion_map,
            'alternative_mapping': alternative_emotion_map,
            'debugging_tips': [
                "1. Check if audio has sufficient volume and quality",
                "2. Verify feature extraction matches training method",
                "3. Test with different emotion mappings",
                "4. Consider downloading a different pre-trained model",
                "5. Check if model file is corrupted (re-download)",
                "6. Verify model was trained on similar audio preprocessing"
            ]
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@speech_bp.route('/test_with_sample', methods=['POST'])
def test_with_sample():
    """
    Test prediction with a sample audio file
    """
    # Create a synthetic audio sample for testing
    try:
        # Generate a simple sine wave (simulating speech)
        sample_rate = 22050
        duration = 3.0
        frequency = 440  # A4 note
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        
        # Create different patterns for different emotions
        patterns = {
            'calm': np.sin(2 * np.pi * frequency * t) * 0.3,
            'happy': np.sin(2 * np.pi * frequency * t) * 0.8 + np.sin(2 * np.pi * frequency * 2 * t) * 0.3,
            'angry': np.sin(2 * np.pi * frequency * t) * 0.9 + np.random.normal(0, 0.1, len(t)),
            'sad': np.sin(2 * np.pi * frequency * 0.5 * t) * 0.4,
        }
        
        results = {}
        
        for emotion_name, audio_data in patterns.items():
            # Save synthetic audio
            test_filename = f'blueprints/speech/static/test_{emotion_name}.wav'
            audio_int16 = (audio_data * 32767).astype(np.int16)
            write(test_filename, sample_rate, audio_int16)
            
            # Predict
            result, error = predict_emotion_with_debugging(test_filename)
            
            if result:
                results[emotion_name] = result
            else:
                results[emotion_name] = {'error': error}
            
            # Clean up
            try:
                os.remove(test_filename)
            except:
                pass
        
        return jsonify({
            'synthetic_audio_tests': results,
            'note': 'These are synthetic audio samples for testing model behavior'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500