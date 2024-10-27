import numpy as np

def predict_fraud(data, model):
    try:
        # Extract features and reshape for model input
        features = np.array([
            data['purchase_value'], 
            data['age'], 
            data['transaction_frequency']  
        ]).reshape(1, -1)
        
        # Make the prediction
        prediction = model.predict(features)[0]
        
        return int(prediction)
    
    except KeyError as e:
        # Raise a more descriptive error if any required field is missing
        missing_field = str(e)
        raise ValueError(f"Missing required field: {missing_field}")
