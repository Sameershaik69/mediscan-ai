import os
import sys
import json
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for

# Dynamic path resolution: Adds the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.predict import Predictor

app = Flask(__name__)

# Configure upload and static folders
UPLOAD_FOLDER = os.path.join('webapp', 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

HISTORY_FILE = os.path.join('webapp', 'static', 'prediction_history.json')

# Initialize inference predictor
MODEL_PATH = os.path.join("model", "saved", "mobilenet_latest.h5")
predictor = Predictor(MODEL_PATH)

def save_to_history(filename, diagnosis, confidence):
    """
    Saves a prediction record to the local JSON history log.
    """
    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "filename": filename,
        "diagnosis": diagnosis,
        "confidence": f"{confidence:.2%}"
    }
    
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                history = json.load(f)
        except Exception:
            history = []
            
    # Add new record to the front of the list and keep last 50 entries
    history.insert(0, record)
    history = history[:50]
    
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
            
        if file:
            # Save original upload
            original_filename = "temp_input.jpeg"
            input_path = os.path.join(app.config['UPLOAD_FOLDER'], original_filename)
            file.save(input_path)
            
            # Setup output filename for Grad-CAM
            gradcam_filename = "temp_gradcam.png"
            output_path = os.path.join(app.config['UPLOAD_FOLDER'], gradcam_filename)
            
            # Run prediction and save Grad-CAM overlay
            diagnosis, confidence = predictor.predict(input_path, save_heatmap_path=output_path)
            
            # Log to history
            save_to_history(file.filename, diagnosis, confidence)
            
            return render_template(
                'index.html', 
                prediction=diagnosis, 
                confidence=f"{confidence:.2%}",
                input_image=original_filename,
                gradcam_image=gradcam_filename
            )
            
    return render_template('index.html')

@app.route('/history')
def history():
    records = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, 'r') as f:
                records = json.load(f)
        except Exception:
            records = []
    return render_template('history.html', records=records)

if __name__ == '__main__':
    # Start the Flask app
    app.run(host='localhost', port=5000, debug=True)