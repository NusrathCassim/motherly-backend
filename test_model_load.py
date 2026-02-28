import os
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# ===============================
# CONFIG
# ===============================
MODEL_PATH = os.path.join("model", "modelv1.keras")
IMAGE_PATH = os.path.join("model", "4.jpg")
IMG_SIZE = 224

CLASS_NAMES = [
    'cutis_marmorata',
    'jaundice',
    'milia',
    'miliaria_heat_rash',
    'mongolian_spots',
    'normal_healthy'
]

# ===============================
# CHECK ENVIRONMENT
# ===============================
print("TensorFlow Version:", tf.__version__)
print("Keras Version:", tf.keras.__version__)
print("GPU Available:", tf.config.list_physical_devices('GPU'))
print("=" * 50)

# ===============================
# CHECK FILES
# ===============================
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model file not found!")

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError("Image file not found!")

print("Model and image found ✅")
print("=" * 50)

# ===============================
# LOAD MODEL
# ===============================
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
print("Model loaded successfully ✅")
print("=" * 50)

# ===============================
# LOAD & PREPROCESS IMAGE
# ===============================
print("Loading image...")
img = Image.open(IMAGE_PATH).convert("RGB")
img = img.resize((IMG_SIZE, IMG_SIZE))

img_array = np.array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = preprocess_input(img_array)

print("Image shape:", img_array.shape)
print("=" * 50)

# ===============================
# PREDICT
# ===============================
print("Making prediction...")
predictions = model.predict(img_array)

predicted_index = np.argmax(predictions[0])
predicted_class = CLASS_NAMES[predicted_index]
confidence = float(predictions[0][predicted_index] * 100)

print("\n🎯 Prediction:", predicted_class)
print(f"📊 Confidence: {confidence:.2f}%")

print("\n🔎 Full Probabilities:")
for i, prob in enumerate(predictions[0]):
    print(f"{CLASS_NAMES[i]}: {prob*100:.2f}%")