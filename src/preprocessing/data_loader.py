import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

def get_class_distribution(data_dir):
    """
    Counts and returns the number of images in each split (train, val, test)
    and class (NORMAL, PNEUMONIA).
    """
    splits = ['train', 'val', 'test']
    classes = ['NORMAL', 'PNEUMONIA']
    distribution = {}

    for split in splits:
        distribution[split] = {}
        for cls in classes:
            path = os.path.join(data_dir, split, cls)
            if os.path.exists(path):
                count = len([f for f in os.listdir(path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                distribution[split][cls] = count
            else:
                distribution[split][cls] = 0
                
    return distribution

def plot_class_distribution(data_dir):
    """
    Plots a bar chart showing the class distribution across splits.
    """
    dist = get_class_distribution(data_dir)
    splits = list(dist.keys())
    
    normal_counts = [dist[split]['NORMAL'] for split in splits]
    pneumonia_counts = [dist[split]['PNEUMONIA'] for split in splits]
    
    x = np.arange(len(splits))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, normal_counts, width, label='NORMAL', color='#4CAF50')
    ax.bar(x + width/2, pneumonia_counts, width, label='PNEUMONIA', color='#FFC107')
    
    ax.set_ylabel('Number of Images')
    ax.set_title('Dataset Distribution')
    ax.set_xticks(x)
    ax.set_xticklabels([s.upper() for s in splits])
    ax.legend()
    plt.tight_layout()
    plt.show()

def plot_sample_images(data_dir, n_per_class=4):
    """
    Plots a grid of sample images from both categories in the train split.
    """
    train_dir = os.path.join(data_dir, 'train')
    classes = ['NORMAL', 'PNEUMONIA']
    
    fig, axes = plt.subplots(len(classes), n_per_class, figsize=(12, 6))
    
    for row, cls in enumerate(classes):
        class_path = os.path.join(train_dir, cls)
        all_images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        sampled_images = random.sample(all_images, n_per_class)
        
        for col, img_name in enumerate(sampled_images):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            ax = axes[row, col]
            ax.imshow(img)
            ax.axis('off')
            if col == 0:
                ax.set_title(cls, fontsize=14, weight='bold', loc='left')
                
    plt.tight_layout()
    plt.show()

def get_data_generators(data_dir, batch_size=32, target_size=(224, 224)):
    """
    Creates and returns ImageDataGenerators for train, val, and test sets.
    """
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    test_dir = os.path.join(data_dir, "test")

    # Training Data Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    # Validation and testing generators use standard scaling only
    val_test_datagen = ImageDataGenerator(rescale=1.0/255.0)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True
    )

    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    return train_generator, val_generator, test_generator

def preprocess_image(image_path, target_size=(224, 224)):
    """
    Preprocesses a single image for inference/Grad-CAM.
    """
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Expand to batch shape (1, 224, 224, 3)
    return img_array