import os
import shutil
import random

# --------------------------
# Paths
# --------------------------
raw_data_dir = "data_raw/PlantVillage"  # point directly to PlantVillage
base_dir = "data"                        # target folder for train/validation
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "validation")

# --------------------------
# Split ratio
# --------------------------
split_ratio = 0.8  # 80% train, 20% validation

# --------------------------
# Remove old train/validation folders if they exist
# --------------------------
for folder in [train_dir, val_dir]:
    if os.path.exists(folder):
        shutil.rmtree(folder)
    os.makedirs(folder, exist_ok=True)

# --------------------------
# Process each class folder
# --------------------------
for class_name in os.listdir(raw_data_dir):
    class_path = os.path.join(raw_data_dir, class_name)
    if not os.path.isdir(class_path):
        continue  # skip if not a folder

    # Gather all image files in the class folder
    images = []
    for root, _, files in os.walk(class_path):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png")):
                images.append(os.path.join(root, file))

    if len(images) == 0:
        print(f"Warning: No images found for class '{class_name}', skipping...")
        continue

    # Shuffle and split
    random.shuffle(images)
    split_index = int(len(images) * split_ratio)
    train_images = images[:split_index]
    val_images = images[split_index:]

    # Create class folders in train/validation
    os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
    os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

    # Copy images to train folder
    for src_path in train_images:
        dst_path = os.path.join(train_dir, class_name, os.path.basename(src_path))
        shutil.copy(src_path, dst_path)

    # Copy images to validation folder
    for src_path in val_images:
        dst_path = os.path.join(val_dir, class_name, os.path.basename(src_path))
        shutil.copy(src_path, dst_path)

    print(f"{class_name}: {len(train_images)} train, {len(val_images)} val images")

print("✅ Dataset split into train/validation successfully!")
