# app.py - Complete Motherly Backend
import os
import datetime
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from PIL import Image
import logging
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try importing tensorflow
try:
    import tensorflow as tf
    from tensorflow.keras.applications.efficientnet import preprocess_input
    TENSORFLOW_AVAILABLE = True
    logger.info(f"✅ TensorFlow {tf.__version__} loaded")
except ImportError as e:
    TENSORFLOW_AVAILABLE = False
    logger.warning(f"⚠️ TensorFlow not available: {e}")

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ===============================
# MODEL CONFIGURATION
# ===============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR,'modelv1.keras') 
IMG_SIZE = 224

CLASS_NAMES = [
    'cutis_marmorata',
    'jaundice',
    'milia',
    'miliaria_heat_rash',
    'mongolian_spots',
    'normal_healthy'
]

# Sinhala translations
SINHALA_NAMES = {
    'cutis_marmorata': 'කැපී පෙනෙන සමේ රටා',
    'jaundice': 'සෙංගමාලය',
    'milia': 'මිලියා',
    'miliaria_heat_rash': 'දද',
    'mongolian_spots': 'මොන්ගෝලියානු ලප',
    'normal_healthy': 'සාමාන්‍ය'
}

# Recommendations (English)
RECOMMENDATIONS_EN = {
    'cutis_marmorata': [
        'Usually harmless in newborns',
        'Keep baby warm',
        'Consult doctor if persistent'
    ],
    'jaundice': [
        'Consult pediatrician immediately',
        'Ensure adequate feeding',
        'Monitor for worsening symptoms'
    ],
    'milia': [
        'No treatment needed',
        'Will clear on its own',
        'Do not squeeze or pick'
    ],
    'miliaria_heat_rash': [
        'Keep baby cool',
        'Dress in light clothing',
        'Use mild soap'
    ],
    'mongolian_spots': [
        'Harmless birthmark',
        'No treatment needed',
        'Usually fades with age'
    ],
    'normal_healthy': [
        'Continue regular care',
        'Maintain feeding schedule',
        'Keep baby clean and dry'
    ]
}

# Recommendations (Sinhala)
RECOMMENDATIONS_SI = {
    'cutis_marmorata': [
        'සාමාන්‍යයෙන් අලුත උපන් බිළිඳුන්ට හානියක් නැත',
        'දරුවා උණුසුම්ව තබා ගන්න',
        'පවතින්නේ නම් වෛද්‍යවරයෙකු හමුවන්න'
    ],
    'jaundice': [
        'වහාම ළමා රෝග විශේෂඥ වෛද්‍යවරයෙකු හමුවන්න',
        'ප්‍රමාණවත් ලෙස පෝෂණය කරන්න',
        'රෝග ලක්ෂණ නරක අතට හැරීම නිරීක්ෂණය කරන්න'
    ],
    'milia': [
        'ප්‍රතිකාර අවශ්‍ය නොවේ',
        'තනිවම පහව යනු ඇත',
        'මිරිකීමෙන් හෝ ඉවත් කිරීමෙන් වළකින්න'
    ],
    'miliaria_heat_rash': [
        'දරුවා සිසිල්ව තබා ගන්න',
        'සැහැල්ලු ඇඳුම් අඳින්න',
        'මෘදු සබන් භාවිතා කරන්න'
    ],
    'mongolian_spots': [
        'හානිකර උපන් ලපයක්',
        'ප්‍රතිකාර අවශ්‍ය නොවේ',
        'වයසත් සමඟ මැකී යයි'
    ],
    'normal_healthy': [
        'නිතිපතා සත්කාර කරන්න',
        'පෝෂණ කාලසටහන පවත්වා ගන්න',
        'දරුවා පිරිසිදුව තබා ගන්න'
    ]
}

# ===============================
# LOAD MODEL (SAME AS TEST SCRIPT)
# ===============================
model = None

if TENSORFLOW_AVAILABLE:
    try:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"❌ Model file not found at: {MODEL_PATH}")
            logger.info(f"📁 Please place your model file at: {MODEL_PATH}")
        else:
            logger.info(f"📂 Loading model from: {MODEL_PATH}")
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            logger.info("✅ Model loaded successfully!")
            
            # Warm up
            dummy_input = np.zeros((1, IMG_SIZE, IMG_SIZE, 3))
            model.predict(dummy_input, verbose=0)
            logger.info("✅ Model warmed up")
            
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        logger.error(traceback.format_exc())
        model = None
else:
    logger.warning("⚠️ Running in demo mode without TensorFlow")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_data):
    """EXACT same preprocessing as your working test script"""
    img = Image.open(image_data).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'name': 'Motherly Backend API',
        'version': '1.0.0',
        'status': 'running',
        'model_loaded': model is not None,
        'tensorflow_available': TENSORFLOW_AVAILABLE,
        'model_path': MODEL_PATH,
        'model_exists': os.path.exists(MODEL_PATH),
        'classes': CLASS_NAMES,
        'endpoints': {
            '/': 'This information',
            '/health': 'Health check',
            '/model-info': 'Model information',
            '/analyze-image': 'POST - Analyze baby image'
        }
    }), 200

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'tensorflow_available': TENSORFLOW_AVAILABLE,
        'timestamp': datetime.datetime.now().isoformat()
    }), 200

@app.route('/model-info', methods=['GET'])
def model_info():
    if model is None:
        return jsonify({
            'error': 'Model not loaded',
            'classes': CLASS_NAMES,
            'sinhala_names': SINHALA_NAMES,
            'demo_mode': True
        }), 200
    
    return jsonify({
        'model_loaded': True,
        'classes': CLASS_NAMES,
        'sinhala_names': SINHALA_NAMES,
        'num_classes': len(CLASS_NAMES),
        'input_shape': f'{IMG_SIZE}x{IMG_SIZE}x3',
        'tensorflow_version': tf.__version__
    }), 200

@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    if model is None:
        return jsonify({
            'error': 'Model not loaded',
            'demo_mode': True,
            'note': 'Model file not found or failed to load'
        }), 503
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed. Use PNG, JPG or JPEG'}), 400
    
    is_sinhala = request.form.get('language', 'en') == 'si'
    
    try:
        logger.info(f"Processing image: {file.filename}")
        
        # Preprocess (exactly like test script)
        img_array = preprocess_image(file)
        
        # Predict (exactly like test script)
        predictions = model.predict(img_array, verbose=0)
        predicted_index = int(np.argmax(predictions[0]))
        confidence = float(predictions[0][predicted_index])
        predicted_class = CLASS_NAMES[predicted_index]
        
        logger.info(f"Prediction: {predicted_class} ({confidence:.2f})")
        
        # Prepare response
        response = {
            'prediction': predicted_class,
            'prediction_sinhala': SINHALA_NAMES.get(predicted_class, predicted_class),
            'confidence': confidence,
            'confidence_percentage': f"{confidence*100:.2f}%",
            'requires_attention': predicted_class == 'jaundice',
            'recommendations': RECOMMENDATIONS_SI[predicted_class] if is_sinhala else RECOMMENDATIONS_EN[predicted_class],
            'all_probabilities': [
                {
                    'class': CLASS_NAMES[i],
                    'class_sinhala': SINHALA_NAMES[CLASS_NAMES[i]],
                    'probability': float(predictions[0][i])
                }
                for i in range(len(CLASS_NAMES))
            ],
            'timestamp': datetime.datetime.now().isoformat()
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/feedback', methods=['POST'])
def feedback():
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    logger.info(f"Feedback received: {data}")
    
    # Save feedback
    try:
        feedback_file = os.path.join(BASE_DIR, 'feedback.log')
        with open(feedback_file, 'a') as f:
            f.write(f"{datetime.datetime.now().isoformat()}: {data}\n")
    except:
        pass
    
    return jsonify({'status': 'success', 'message': 'Feedback received. Thank you!'}), 200

@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large. Max size is 16MB'}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("\n" + "="*60)
    print("MOTHERLY BACKEND SERVER")
    print("="*60)
    print(f" Python version: {os.sys.version}")
    print(f" TensorFlow available: {TENSORFLOW_AVAILABLE}")
    print(f" TensorFlow version: {tf.__version__ if TENSORFLOW_AVAILABLE else 'N/A'}")
    print(f" Model path: {MODEL_PATH}")
    print(f" Model exists: {os.path.exists(MODEL_PATH)}")
    print(f" Model loaded: {model is not None}")
    if model is None:
        print("  MODEL NOT LOADED! ")
        print(f" Please place your model file at: {MODEL_PATH}")
        print(f" Make sure the file name is: motherly_model_final.h5")
    else:
        print(f" Model loaded successfully!")
    print(f" Server URL: http://localhost:5000")
    print(f" Press CTRL+C to stop")
    print("="*60 + "\n")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )