# 🩻 MediScan AI — AI-Based Medical Image Diagnosis System

An end-to-end deep learning project that classifies chest X-rays using transfer learning (MobileNetV2), provides visual explainability using Grad-CAM, and is deployed as a live full-stack web application.

---

## 🎯 Project Objectives
- Build a deep learning model to classify chest X-rays as **NORMAL** or **PNEUMONIA**
- Utilize MobileNetV2 Transfer Learning for high classification accuracy
- Implement Grad-CAM to visualize which lung regions influenced the AI's prediction
- Deploy as an interactive, live Streamlit web application

---

## 📁 Project Structure
```text
mediscan-ai/
├── dataset/
│   └── raw/              ← Raw train/val/test splits (not uploaded to GitHub)
├── model/
│   └── saved/            ← Trained weights (mobilenet_latest.h5)
├── src/
│   ├── preprocessing/
│   │   └── data_loader.py    ← Dataset loading, analytics, and augmentation
│   ├── training/
│   │   └── model.py          ← Custom CNN & MobileNetV2 structures
│   ├── evaluation/
│   │   └── evaluate.py       ← Metrics calculation and plotting
│   ├── gradcam/
│   │   └── gradcam.py        ← Grad-CAM explainability implementation
│   └── predict.py            ← Offline inference engine
├── webapp/
│   ├── app.py                ← Flask server application
│   └── templates/
│       ├── index.html        ← Flask diagnostic interface
│       └── history.html      ← Flask prediction log
├── notebooks/
│   └── training_notebook.ipynb  ← Interactive training walkthrough
├── streamlit_app.py          ← Streamlit web application frontend
├── requirements.txt          ← System dependencies
└── runtime.txt               ← Python version  

---
## 🚀 Live Deployment

The application is deployed live on the web and can be accessed at:
* **Streamlit App**: [https://mediscan-ai.streamlit.app/](https://mediscan-ai.streamlit.app/) *(You can replace this with your actual live URL once the deployment completes)*

---

## 📦 How to Run Locally

### 1. Set Up Environment
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate

pip install -r requirements.txt