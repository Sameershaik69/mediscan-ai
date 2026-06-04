import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from src.preprocessing.data_loader import get_data_generators

def evaluate_model(model_path, data_dir):
    """
    Loads a trained model, evaluates it on the test set,
    and returns metrics alongside raw prediction arrays.
    """
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)
    
    # Load generators (only test_generator is needed here)
    _, _, test_gen = get_data_generators(data_dir)
    
    print("Evaluating model on test set...")
    # Predict probabilities
    y_prob = model.predict(test_gen, verbose=1)
    
    # Class labels
    y_pred = np.argmax(y_prob, axis=1)
    y_true = test_gen.classes
    
    # Calculate Metrics
    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred, target_names=test_gen.class_indices.keys(), output_dict=True)
    auc = roc_auc_score(y_true, y_prob[:, 1])
    
    print("\n--- Test Metrics Evaluation ---")
    print(classification_report(y_true, y_pred, target_names=test_gen.class_indices.keys()))
    print(f"ROC AUC Score: {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    metrics = {
        "confusion_matrix": cm,
        "classification_report": cr,
        "auc_score": auc
    }
    
    return metrics, y_true, y_pred, y_prob

def plot_training_history(history):
    """
    Helper function to plot training accuracy and loss curves.
    """
    acc = history.history.get('accuracy') or history.history.get('categorical_accuracy', [])
    val_acc = history.history.get('val_accuracy') or history.history.get('val_categorical_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])
    
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(12, 5))
    
    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')
    
    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    
    plt.tight_layout()
    plt.show()