# main.py - Flask REST API for Rent Prediction

from flask import Flask, request, jsonify
import pandas as pd
from catboost import CatBoostRegressor
from flask_cors import CORS

MODEL_PATH = "models/catboost_model.cbm"

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable Flutter/Web access

# Load model
model = CatBoostRegressor()
model.load_model(MODEL_PATH)

@app.route("/")
def home():
    return jsonify({"message": "Rent Prediction API is running"})


@app.route("/predict", methods=["POST"])
def predict_rent():
    try:
        data = request.get_json()

        # Required fields
        required_fields = ["Bed", "Bath", "Month", "SubArea", "Region", "PropertyType"]

        # Validate input
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        # Create DataFrame for prediction
        user_df = pd.DataFrame([{
            "Bed": int(data["Bed"]),
            "Bath": int(data["Bath"]),
            "Month": int(data["Month"]),
            "SubArea": data["SubArea"],
            "Region": data["Region"],
            "PropertyType": data["PropertyType"]
        }])

        # Predict
        predicted_rent = model.predict(user_df)[0]

        return jsonify({
            "status": "success",
            "predicted_rent": round(float(predicted_rent), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("🔥 Flask API Running on http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
