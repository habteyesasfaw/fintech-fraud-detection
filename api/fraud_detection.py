# fraud_detection.py

import numpy as np

def predict_fraud(data, model):
    features = np.array([
        data['purchase_value'], 
        data['age'], 
        data['transaction_frequency']  # Example feature names
    ]).reshape(1, -1)
    
    prediction = model.predict(features)[0]
    return int(prediction)
