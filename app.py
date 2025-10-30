from flask import Flask, render_template, request, jsonify
import requests
import pickle
from io import BytesIO
import numpy as np
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# 🔗 Load model from Hugging Face
MODEL_URL = "https://huggingface.co/Palak-Zade/crop-recommender-model/resolve/main/crop_model.pkl"

print("\n🔄 Downloading model from Hugging Face...")
response = requests.get(MODEL_URL)

if response.status_code == 200:
    package = pickle.load(BytesIO(response.content))
    model = package['model']
    le = package['label_encoder']
    model_name = package['model_name']
    accuracy = package['accuracy']
    crops = package['crops']
    print(f"✅ Model loaded: {model_name}")
else:
    print("❌ Failed to download model from Hugging Face!")
    model = le = model_name = accuracy = crops = None


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        nitrogen = float(data['nitrogen'])
        phosphorus = float(data['phosphorus'])
        potassium = float(data['potassium'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph_val = float(data['ph'])
        rainfall = float(data['rainfall'])

        # Validation
        if not (0 <= ph_val <= 14):
            return jsonify({'success': False, 'error': 'pH must be 0-14'}), 400
        if not (0 <= humidity <= 100):
            return jsonify({'success': False, 'error': 'Humidity must be 0-100'}), 400
        if rainfall < 0:
            return jsonify({'success': False, 'error': 'Rainfall cannot be negative'}), 400

        # Prediction
        features = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, ph_val, rainfall]])
        prediction = model.predict(features)
        predicted_crop = le.inverse_transform(prediction)[0]

        confidence = None
        top_predictions = None

        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0]
            confidence = float(np.max(proba) * 100)
            top_indices = np.argsort(proba)[-3:][::-1]
            top_predictions = [
                {'crop': le.inverse_transform([idx])[0], 'probability': float(proba[idx] * 100)}
                for idx in top_indices
            ]

        return jsonify({
            'success': True,
            'crop': predicted_crop,
            'confidence': confidence,
            'top_predictions': top_predictions,
            'model_info': {'name': model_name, 'accuracy': accuracy}
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/model-info', methods=['GET'])
def model_info():
    if model is not None:
        return jsonify({'success': True, 'model_name': model_name, 'accuracy': accuracy, 'crops': crops})
    else:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 404


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})


# ✅ For Vercel deployment
app = app
