import os
import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# --------------------------
# Paths to saved model and class indices
# --------------------------
MODEL_PATH = "tiny_plant_disease_model.h5"
CLASS_INDICES_PATH = "class_indices.json"
IMG_SIZE = 64  # must match training input size

# --------------------------
# Load model
# --------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# --------------------------
# Load class indices
# --------------------------
if not os.path.exists(CLASS_INDICES_PATH):
    raise FileNotFoundError(f"Class indices file not found: {CLASS_INDICES_PATH}")

with open(CLASS_INDICES_PATH, "r") as f:
    class_indices = json.load(f)

# Reverse mapping: index -> class name
idx_to_class = {v: k for k, v in class_indices.items()}

# --------------------------
# Prediction function
# --------------------------
def predict_disease(img_path):
    if not os.path.exists(img_path):
        print(f"Image not found: {img_path}")
        return
    
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    predictions = model.predict(img_array)
    predicted_idx = np.argmax(predictions[0])
    confidence = predictions[0][predicted_idx]

    # Output
    predicted_class = idx_to_class[predicted_idx]
    print(f"Predicted disease: {predicted_class} | Confidence: {confidence:.2f}")

# --------------------------
# Main loop for repeated predictions
# --------------------------
if __name__ == "__main__":
    while True:
        img_path = input("Enter path to leaf image (or 'exit' to quit): ").strip()
        if img_path.lower() == "exit":
            break
        predict_disease(img_path)
