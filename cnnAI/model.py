import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import numpy as np

# Load Pre-trained Model (MobileNetV2)
model = MobileNetV2(weights="imagenet")

def classify_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to MobileNetV2 input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)

    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]  # Get top 3 predictions

    results = [{"label": label, "confidence": float(confidence)} for _, label, confidence in decoded_predictions]
    return results
