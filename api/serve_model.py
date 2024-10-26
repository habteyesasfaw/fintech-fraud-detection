# serve_model.py

from flask import Flask, request, jsonify
from fraud_detection import predict_fraud
from utils import load_model, initialize_logger

app = Flask(__name__)

# Initialize logger
logger = initialize_logger()

# Load model
model = load_model('../models/fraud_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        prediction = predict_fraud(data, model)
        response = {"prediction": prediction}
        logger.info(f"Request: {data}, Prediction: {prediction}")
        return jsonify(response)
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
