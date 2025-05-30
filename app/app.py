import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from flask import Flask, request, jsonify
import joblib
import pandas as pd
from preprocess import preprocess

app = Flask(__name__)

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), "../src/model.pkl")
model = joblib.load(model_path)

@app.route("/")
def home():
    return "✅ Model API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        df = preprocess(df)
        prediction = model.predict(df)[0]
        return jsonify({"prediction": int(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
