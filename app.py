from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Load the single model package
try:
    print("\n🔄 Loading model package...")
    package = joblib.load('crop_model.pkl')
    
    model = package['model']
    le = package['label_encoder']
    model_name = package['model_name']
    accuracy = package['accuracy']
    crops = package['crops']
    
    print(f"✅ Model loaded: {model_name}")
    print(f"✅ Accuracy: {accuracy:.4f}")
    print(f"✅ Crops: {len(crops)}")
    
except FileNotFoundError:
    print("❌ ERROR: crop_model.pkl not found!")
    print("📝 Please run: python simple_train_model.py")
    model = None
    le = None
    model_name = None
    accuracy = None
    crops = None
    
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    le = None
    model_name = None
    accuracy = None
    crops = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None or le is None:
            return jsonify({
                'success': False,
                'error': 'Model not loaded. Please run simple_train_model.py first.'
            }), 500
        
        # Get data from request
        data = request.get_json()
        
        # Extract features
        nitrogen = float(data['nitrogen'])
        phosphorus = float(data['phosphorus'])
        potassium = float(data['potassium'])
        temperature = float(data['temperature'])
        humidity = float(data['humidity'])
        ph = float(data['ph'])
        rainfall = float(data['rainfall'])
        
        # Validate ranges
        if not (0 <= ph <= 14):
            return jsonify({'success': False, 'error': 'pH must be 0-14'}), 400
        
        if not (0 <= humidity <= 100):
            return jsonify({'success': False, 'error': 'Humidity must be 0-100'}), 400
        
        if rainfall < 0:
            return jsonify({'success': False, 'error': 'Rainfall cannot be negative'}), 400
        
        # Create feature array
        features = np.array([[nitrogen, phosphorus, potassium, temperature,
                            humidity, ph, rainfall]])
        
        # Make prediction
        prediction = model.predict(features)
        predicted_crop = le.inverse_transform(prediction)[0]
        
        # Get probabilities if available
        confidence = None
        top_predictions = None
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(features)[0]
            confidence = float(np.max(proba) * 100)
            
            # Top 3 predictions
            top_indices = np.argsort(proba)[-3:][::-1]
            top_predictions = [
                {
                    'crop': le.inverse_transform([idx])[0],
                    'probability': float(proba[idx] * 100)
                }
                for idx in top_indices
            ]
        
        return jsonify({
            'success': True,
            'crop': predicted_crop,
            'confidence': confidence,
            'top_predictions': top_predictions,
            'model_info': {
                'name': model_name,
                'accuracy': accuracy
            }
        })
    
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error: {str(e)}'
        }), 500

@app.route('/model-info', methods=['GET'])
def model_info():
    if model is not None:
        return jsonify({
            'success': True,
            'model_name': model_name,
            'accuracy': accuracy,
            'crops': crops
        })
    else:
        return jsonify({'success': False, 'error': 'Model not loaded'}), 404

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load your model
model = pickle.load(open("crop_model.pkl", "rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # your prediction logic here
    return "Prediction result"

if __name__ == '__main__':
    app.run()

