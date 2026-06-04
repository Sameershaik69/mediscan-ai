import os
import tensorflow as tf

print("=== Environment Verification ===")
print(f"TensorFlow Version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}\n")

# Define paths to check
base_dir = os.path.join("dataset", "raw")
splits = ["train", "val", "test"]
classes = ["NORMAL", "PNEUMONIA"]

print("=== Dataset Verification ===")
all_ok = True

for split in splits:
    split_path = os.path.join(base_dir, split)
    if not os.path.exists(split_path):
        print(f"[ERROR] Missing folder: {split_path}")
        all_ok = False
        continue
    
    print(f"\nChecking split: '{split}'")
    for cls in classes:
        class_path = os.path.join(split_path, cls)
        if not os.path.exists(class_path):
            print(f"  [ERROR] Missing class folder: {class_path}")
            all_ok = False
        else:
            # Count images in the folder
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            print(f"  [OK] Found {len(images)} images in {cls}")

print("\n================================")
if all_ok:
    print("Verification Successful! You are ready to start writing the model code.")
else:
    print("Verification failed. Please check the folder paths above.")