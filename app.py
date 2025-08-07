import os
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
# NEW: Import the TFLite runtime interpreter
import tflite_runtime.interpreter as tflite
from PIL import Image
import numpy as np
from scipy.stats import norm

# --- Configuration ---
TARGET_SIZE = (150, 150) 
CLASS_NAMES = ['Healthy', 'Late Blight']
# NEW: Point to the optimized .tflite model
MODEL_PATH = 'leaf_model.tflite'
UPLOAD_FOLDER = 'static/uploads'

# Initialize the Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- NEW: Load the TFLite Model ---
try:
    interpreter = tflite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("--- TFLite Model loaded successfully! ---")
except Exception as e:
    print(f"FATAL: Could not load TFLite model. Error: {e}")
    interpreter = None

# --- Preprocessing Function (no changes needed) ---
def preprocess_image(image_path, target_size):
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize(target_size)
        img_array = np.array(img, dtype=np.float32) # TFLite expects float32
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        return img_array
    except Exception as e:
        print(f"Error preprocessing image {image_path}: {e}")
        return None

# --- Flask Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if interpreter is None:
        return "Model not loaded. Please check server logs.", 500

    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        processed_image = preprocess_image(filepath, TARGET_SIZE)
        if processed_image is None:
            return "Error processing the uploaded image.", 500

        try:
            # --- NEW: Prediction with TFLite Interpreter ---
            interpreter.set_tensor(input_details[0]['index'], processed_image)
            interpreter.invoke()
            prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]
            
            if prediction > 0.5:
                result = CLASS_NAMES[1]
                confidence = prediction
            else:
                result = CLASS_NAMES[0]
                confidence = 1 - prediction

            z_score = (confidence - 0.5) / (np.sqrt(0.5 * (1 - 0.5) / 1))
            p_value = 1 - norm.cdf(abs(z_score))
            
            template_image_path = os.path.join('uploads', filename).replace('\\', '/')

            return render_template('result.html', 
                                   prediction_text=result, 
                                   image_path=template_image_path,
                                   confidence=confidence,
                                   z_score=z_score,
                                   p_value=p_value)

        except Exception as e:
            print(f"Error during prediction: {e}")
            raise e

    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
