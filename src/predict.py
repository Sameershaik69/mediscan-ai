import os
import sys

# Dynamic path resolution: Adds the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from tensorflow.keras.models import load_model
from src.preprocessing.data_loader import preprocess_image
from src.gradcam.gradcam import GradCAM

class Predictor:
    def __init__(self, model_path, class_names=None):
        """
        Initializes the predictor by loading the saved model.
        """
        if class_names is None:
            self.class_names = ['NORMAL', 'PNEUMONIA']
        else:
            self.class_names = class_names
            
        print(f"Loading model for inference from: {model_path}")
        self.model = load_model(model_path)
        self.gradcam = GradCAM(self.model)

    def predict(self, image_path, save_heatmap_path=None):
        """
        Runs inference on a single image.
        Returns: predicted_class, confidence
        """
        # Preprocess the image for the model
        img_array = preprocess_image(image_path)
        
        # Run prediction
        predictions = self.model.predict(img_array)
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        predicted_class = self.class_names[class_idx]
        
        # Generate Grad-CAM if a save path is provided
        if save_heatmap_path:
            # Read original image
            original_np = cv2.imread(image_path)
            original_np = cv2.cvtColor(original_np, cv2.COLOR_BGR2RGB)
            original_np = cv2.resize(original_np, (224, 224))
            
            # Generate the visualization
            self.gradcam.visualize(
                img_array, 
                original_np, 
                class_names=self.class_names,
                save_path=save_heatmap_path
            )
            
        return predicted_class, confidence

# Quick test execution
if __name__ == "__main__":
    MODEL_PATH = os.path.join("model", "saved", "mobilenet_latest.h5")
    
    # Check if model exists before running test
    if os.path.exists(MODEL_PATH):
        predictor = Predictor(MODEL_PATH)
        print("Inference engine initialized successfully.")
    else:
        print(f"Model file not found at: {MODEL_PATH}. Please train the model first.")