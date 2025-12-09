Plant Disease Detection Backend

This backend module provides a plant leaf disease detection system using a Tiny CNN model trained on the PlantVillage dataset. The system can predict diseases for Potato, Tomato, and Pepper leaves.

Project Overview

The project includes:

Dataset Preparation

Raw images are stored in data\_raw/PlantVillage.

prepare\_split.py automatically splits the dataset into training and validation sets.

Supports all classes in the dataset, including Potato, Tomato, and Pepper.

Example split command output:

Potato\_\_\_Early\_blight: 800 train, 200 val images
Tomato\_Early\_blight: 800 train, 200 val images
Pepper\_\_bell\_\_\_Bacterial\_spot: 797 train, 200 val images



The script counts the images per class and ensures balanced splits.

Model Training

tiny\_cnn.py defines a small CNN for plant disease classification.

Uses Conv2D, MaxPooling2D, Flatten, and Dense layers.

Training images are normalized and augmented using ImageDataGenerator.

Validation split is also handled during training.

After training:

Model saved as tiny\_plant\_disease\_model.h5.

Class indices saved as class\_indices.json.

Prediction

test\_prediction.py allows testing on a single leaf image.

Loads the trained model and class indices.

Outputs predicted disease and confidence score.

Example output:

Predicted disease: Tomato\_Early\_blight | Confidence: 0.97



Normalizes input images to match training settings (64x64 RGB).

Backend Integration (planned)

plant\_disease\_detector.py wraps the model loading and prediction into a reusable function.

Ready to integrate with FastAPI or Flask endpoints for frontend upload and predictions.

Folder Structure
backend/
│
├─ api/
│   ├─ plant\_disease\_detector.py  # Model loader and prediction function
│   ├─ test\_prediction.py         # Command-line testing script
│   ├─ tiny\_cnn.py                # Model training script
│   ├─ prepare\_split.py           # Dataset split script
│
data\_raw/PlantVillage/            # Raw dataset (Potato, Tomato, Pepper)
data/train/                        # Training images after split
data/validation/                   # Validation images after split
test\_images/                       # Example leaf images for testing
tiny\_plant\_disease\_model.h5        # Trained CNN model
class\_indices.json                 # Class label mapping

How to Use

Prepare Dataset

python prepare\_split.py



Automatically splits raw dataset into train and validation folders.

Prints number of images per class.

Train Model

python tiny\_cnn.py --epochs 10 --batch\_size 16



Trains the CNN on the prepared dataset.

Saves model and class indices.

Test Prediction

python test\_prediction.py



Enter the path to a leaf image when prompted.

Example input: test\_images/leaf1.webp

Output shows predicted disease and confidence score.

