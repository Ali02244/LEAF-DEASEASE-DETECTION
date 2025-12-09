import os
import json
import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# --------------------------
# 0. Argument parsing
# --------------------------
parser = argparse.ArgumentParser(description="Train a Tiny CNN for Plant Disease Classification")
parser.add_argument('--epochs', type=int, default=15, help='Number of epochs for training')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
args = parser.parse_args()

# --------------------------
# 1. Dataset paths
# --------------------------
train_data_dir = 'data/train'
val_data_dir = 'data/validation'

if not os.path.exists(train_data_dir):
    raise FileNotFoundError(f"Training data folder not found: {train_data_dir}")
if not os.path.exists(val_data_dir):
    raise FileNotFoundError(f"Validation data folder not found: {val_data_dir}")

# --------------------------
# 2. Data preprocessing & augmentation
# --------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.1,
    shear_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(64, 64),
    batch_size=args.batch_size,
    class_mode='categorical',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=(64, 64),
    batch_size=args.batch_size,
    class_mode='categorical',
    shuffle=False
)

print("Detected classes:", train_generator.class_indices)

# --------------------------
# 3. Tiny CNN model
# --------------------------
model = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(64,64,3)),
    MaxPooling2D(2,2),

    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(128, activation='relu'),
    Dense(train_generator.num_classes, activation='softmax')
])

# --------------------------
# 4. Compile model
# --------------------------
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --------------------------
# 5. Callbacks
# --------------------------
checkpoint = ModelCheckpoint("tiny_plant_disease_model.h5", save_best_only=True, monitor="val_accuracy")
earlystop = EarlyStopping(patience=5, restore_best_weights=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

# --------------------------
# 6. Train model
# --------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=args.epochs,
    callbacks=[checkpoint, earlystop, lr_reduce],
    verbose=1
)

# --------------------------
# 7. Save class indices
# --------------------------
class_indices_path = "class_indices.json"
with open(class_indices_path, "w") as f:
    json.dump(train_generator.class_indices, f)
print(f"Class indices saved as {class_indices_path}")

# --------------------------
# 8. Plot training history
# --------------------------
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title("Training Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

plt.figure(figsize=(8,5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
