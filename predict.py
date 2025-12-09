import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Load trained model
model = load_model("tiny_plant_disease_model.h5")

# Map class indices to folder names
class_indices = {'Healthy': 0, 'Tomato_Early_Blight': 2, 'Tomato_Late_Blight': 1}
class_labels = {v: k for k, v in class_indices.items()}

# Function to predict disease from image
def predict_disease(img_path):
    if not os.path.exists(img_path):
        print("Image does not exist!")
        return
    
    img = image.load_img(img_path, target_size=(64,64))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    pred = model.predict(img_array)
    pred_class = np.argmax(pred)
    print(f"Predicted Class: {class_labels[pred_class]}")
    
# Example usage
predict_disease("data/train/Healthy/leaf1.jpg")
