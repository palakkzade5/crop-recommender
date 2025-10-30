from flask import Flask, render_template, request, jsonify
import requests
import pickle
from io import BytesIO
import numpy as np
from flask_cors import CORS
import traceback
import sys

app = Flask(__name__, template_folder="../templates")
CORS(app)

# ✅ Model stored remotely on Hugging Face
MODEL_URL = "https://huggingface.co/Palak-Zade/crop-recommender-model/resolve/main/crop_model.pkl"

# Global variables for lazy loading
model, le, model_name, accuracy, crops = None, None, None, None, None
model_load_error = None


# ✅ Function to load model only when needed
def load_model():
    global model, le, model_name, accuracy, crops, model_load_error
    if model is None:
        print("🔄 Downloading model from Hugging Face...", file=sys.stderr)
        print(f"🔗 URL: {MODEL_URL}", file=sys.stderr)
        try:
            response = requests.get(MODEL_URL, timeout=60)
            print(f"📡 Response status code: {response.status_code}", file=sys.stderr)
            print(f"📦 Content length: {len(response.content)} bytes", file=sys.stderr)
            
            if response.status_code == 200:
                try:
                    package = pickle.load(BytesIO(response.content))
                    print(f"📋 Package keys: {package.keys()}", file=sys.stderr)
                    
                    model = package["model"]
                    le = package["label_encoder"]
                    model_name = package.get("model_name", "Naive Bayes")
                    accuracy = package.get("accuracy", "N/A")
                    crops = package.get("crops", [])
                    
                    print(f"✅ Model loaded: {model_name} (Accuracy: {accuracy}%)", file=sys.stderr)
                    print(f"🌾 Available crops: {len(crops) if crops else 'N/A'}", file=sys.stderr)
                    return True
                except Exception as e:
                    error_msg = f"Failed to unpickle model: {str(e)}"
                    print(f"❌ {error_msg}", file=sys.stderr)
                    print(traceback.format_exc(), file=sys.stderr)
                    model_load_error = error_msg
                    return False
            else:
                error_msg = f"Failed to download model. Status code: {response.status_code}"
                print(f"❌ {error_msg}", file=sys.stderr)
                print(f"Response text: {response.text[:500]}", file=sys.stderr)
                model_load_error = error_msg
                return False
                
        except requests.exceptions.Timeout:
            error_msg = "Request timed out while downloading model"
            print(f"❌ {error_msg}", file=sys.stderr)
            model_load_error = error_msg
            return False
        except requests.exceptions.RequestException as e:
            error_msg = f"Network error: {str(e)}"
            print(f"❌ {error_msg}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            model_load_error = error_msg
            return False
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            print(f"❌ {error_msg}", file=sys.stderr)
            print(traceback.format_exc(), file=sys.stderr)
            model_load_error = error_msg
            return False
    return True


# ✅ Home route
@app.route("/")
def home():
    return render_template("index.html")


# ✅ Predict route
@app.route("/predict", methods=["POST"])
def predict():
    print("=" * 50, file=sys.stderr)
    print("🚀 Prediction request received", file=sys.stderr)
    
    try:
        # Load model
        if not load_model():
            error_message = model_load_error or "Model failed to load from Hugging Face"
            print(f"❌ Returning error: {error_message}", file=sys.stderr)
            return jsonify({
                "success": False,
                "error": error_message
            }), 500

        # 🛑 Check if model failed to load
        if model is None:
            print("❌ Model is None after load attempt", file=sys.stderr)
            return jsonify({
                "success": False,
                "error": "Model is not available"
            }), 500

        data = request.get_json()
        print(f"📩 Received data: {data}", file=sys.stderr)

        if not data:
            print("❌ No input data provided", file=sys.stderr)
            return jsonify({
                "success": False,
                "error": "No input data provided"
            }), 400

        # Check all required keys
        required_keys = ["nitrogen", "phosphorus", "potassium", "temperature", "humidity", "ph", "rainfall"]
        missing_keys = [key for key in required_keys if key not in data]
        
        if missing_keys:
            print(f"❌ Missing keys: {missing_keys}", file=sys.stderr)
            return jsonify({
                "success": False,
                "error": f"Missing keys: {', '.join(missing_keys)}"
            }), 400

        # Convert input to numpy array
        try:
            features = np.array([
                float(data["nitrogen"]),
                float(data["phosphorus"]),
                float(data["potassium"]),
                float(data["temperature"]),
                float(data["humidity"]),
                float(data["ph"]),
                float(data["rainfall"])
            ]).reshape(1, -1)
            print(f"🔢 Features array: {features}", file=sys.stderr)
        except (ValueError, TypeError) as e:
            print(f"❌ Invalid numeric values: {e}", file=sys.stderr)
            return jsonify({
                "success": False,
                "error": f"Invalid numeric values: {str(e)}"
            }), 400

        # Make prediction
        print("🤖 Making prediction...", file=sys.stderr)
        prediction = model.predict(features)
        print(f"📊 Raw prediction: {prediction}", file=sys.stderr)
        
        predicted_crop = le.inverse_transform(prediction)[0]
        print(f"🌾 Predicted crop: {predicted_crop}", file=sys.stderr)

        result = {
            "success": True,
            "crop": predicted_crop,
            "model_name": model_name,
            "accuracy": accuracy
        }

        print(f"✅ Returning result: {result}", file=sys.stderr)
        return jsonify(result), 200

    except Exception as e:
        error_msg = f"Server error: {str(e)}"
        print(f"❌ {error_msg}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        return jsonify({
            "success": False,
            "error": error_msg
        }), 500


# ✅ Health check route
@app.route("/health")
def health():
    model_status = "loaded" if model is not None else "not loaded"
    return jsonify({
        "status": "ok",
        "model_status": model_status,
        "model_url": MODEL_URL,
        "model_load_error": model_load_error
    }), 200


# For local development
if __name__ == "__main__":
    app.run(debug=True)