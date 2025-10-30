from flask import Flask, render_template, request, jsonify
import requests
import pickle
from io import BytesIO
import numpy as np
from flask_cors import CORS

app = Flask(__name__, template_folder="../templates")
CORS(app)

# ✅ Model stored remotely on Hugging Face
MODEL_URL = "https://huggingface.co/Palak-Zade/crop-recommender-model/resolve/main/crop_model.pkl"

# Global variables for lazy loading
model, le, model_name, accuracy, crops = None, None, None, None, None


# ✅ Function to load model only when needed
def load_model():
    global model, le, model_name, accuracy, crops
    if model is None:
        print("🔄 Downloading model from Hugging Face...")
        try:
            response = requests.get(MODEL_URL)
            if response.status_code == 200:
                package = pickle.load(BytesIO(response.content))
                model = package["model"]
                le = package["label_encoder"]
                model_name = package["model_name"]
                accuracy = package["accuracy"]
                crops = package["crops"]
                print(f"✅ Model loaded: {model_name} (Accuracy: {accuracy}%)")
            else:
                print("❌ Failed to load model from Hugging Face:", response.status_code)
        except Exception as e:
            print("❌ Exception while loading model:", e)


# ✅ Home route
@app.route("/")
def home():
    return render_template("index.html")


# ✅ Predict route
@app.route("/predict", methods=["POST"])
def predict():
    load_model()  # Load only when first requested

    # 🛑 Check if model failed to load
    if model is None:
        return jsonify({"error": "Model failed to load from Hugging Face"}), 500

    try:
        data = request.get_json()
        print("📩 Received data:", data)

        if not data:
            return jsonify({"error": "No input data provided"}), 400

        # Check all required keys
        required_keys = ["nitrogen", "phosphorus", "potassium", "temperature", "humidity", "ph", "rainfall"]
        for key in required_keys:
            if key not in data:
                return jsonify({"error": f"Missing key: {key}"}), 400

        # Convert input to numpy array
        features = np.array([
            data["nitrogen"],
            data["phosphorus"],
            data["potassium"],
            data["temperature"],
            data["humidity"],
            data["ph"],
            data["rainfall"]
        ]).reshape(1, -1)

        # Make prediction
        prediction = model.predict(features)
        predicted_crop = le.inverse_transform(prediction)[0]

        result = {
            "success": True,
            "crop": predicted_crop,
            "model_name": model_name,
            "accuracy": accuracy
        }

        print("✅ Prediction result:", result)
        return jsonify(result)

    except Exception as e:
        print("❌ Error during prediction:", e)
        return jsonify({"error": str(e)}), 500


# ✅ Health check route (for Vercel verification)
@app.route("/health")
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)
