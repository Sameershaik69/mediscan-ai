import os  
import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

class GradCAM:
    def __init__(self, model, layer_name=None):
        """
        Initializes Grad-CAM. If layer_name is None, it tries to find the last conv layer automatically.
        """
        self.model = model
        self.layer_name = layer_name
        
        if self.layer_name is None:
            self.layer_name = self.find_last_conv_layer()

    def find_last_conv_layer(self):
        """
        Walks backward through model layers to find the name of the last convolutional layer.
        """
        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D) or 'conv' in layer.name.lower():
                return layer.name
        raise ValueError("Could not find any convolutional layer in the model.")

    def compute_heatmap(self, img_array, class_index=None):
        """
        Computes the Grad-CAM activation heatmap for a specific input array.
        """
        grad_model = tf.keras.models.Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layer_name).output, self.model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            if class_index is None:
                class_index = tf.argmax(predictions[0])
            loss = predictions[:, class_index]

        grads = tape.gradient(loss, conv_outputs)

        cast_conv_outputs = tf.cast(conv_outputs > 0, "float32")
        cast_grads = tf.cast(grads > 0, "float32")
        guided_grads = cast_conv_outputs * cast_grads * grads

        weights = tf.reduce_mean(guided_grads, axis=(0, 1, 2))

        cam = tf.reduce_sum(tf.multiply(weights, conv_outputs), axis=-1)
        
        heatmap = cam[0].numpy()
        heatmap = np.maximum(heatmap, 0)
        denom = (np.max(heatmap) - np.min(heatmap)) + 1e-10
        heatmap = (heatmap - np.min(heatmap)) / denom
        
        return heatmap

    def visualize(self, img_array, original_np, class_names=None, save_path=None):
        """
        Generates the combined visualization overlay and returns a matplotlib figure.
        """
        predictions = self.model.predict(img_array)
        class_idx = np.argmax(predictions[0])
        confidence = predictions[0][class_idx]
        
        heatmap = self.compute_heatmap(img_array, class_idx)
        
        heatmap_resized = cv2.resize(heatmap, (original_np.shape[1], original_np.shape[0]))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        
        superimposed_img = cv2.addWeighted(original_np, 0.6, heatmap_color, 0.4, 0)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original_np)
        axes[0].set_title("Original Chest X-Ray")
        axes[0].axis("off")
        
        axes[1].imshow(heatmap_resized, cmap='jet')
        axes[1].set_title("Activation Heatmap")
        axes[1].axis("off")
        
        axes[2].imshow(superimposed_img)
        pred_label = class_names[class_idx] if class_names else f"Class {class_idx}"
        axes[2].set_title(f"Grad-CAM Prediction: {pred_label} ({confidence:.2%})")
        axes[2].axis("off")
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Saved Grad-CAM visualization to: {save_path}")
            
        return fig