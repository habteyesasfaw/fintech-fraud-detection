from flask import Flask, request, jsonify
from fraud_detection import predict_fraud
from utils import load_model, initialize_logger

app = Flask(__name__)

# Initialize logger
logger = initialize_logger()

# Load model
model = load_model('models/fraud_model.pkl')  # Ensure the model file path is correct

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse JSON input data
        data = request.get_json()
        
        # Predict fraud
        prediction = predict_fraud(data, model)
        
        # Prepare and return response
        response = {"prediction": prediction}
        logger.info(f"Request: {data}, Prediction: {prediction}")
        return jsonify(response), 200

    except ValueError as ve:
        # Handle missing fields or invalid input
        logger.error(f"Value error: {ve}")
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        # Handle all other errors
        logger.error(f"Error processing request: {e}")
        return jsonify({"error": "An unexpected error occurred"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
