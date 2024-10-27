# utils.py

import joblib
import logging

def load_model(model_path):
    return joblib.load(model_path)

def initialize_logger():
    logger = logging.getLogger('fraud_detections')
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler('../logs/app.log')
    handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s'))
    logger.addHandler(handler)
    return logger
