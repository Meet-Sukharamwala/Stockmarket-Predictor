# src/model_utils.py
import joblib
import os
import tensorflow as tf
from src.utils.logger import get_logger

logger = get_logger(__name__)

def save_sklearn_model(model, name: str, directory: str = "models"):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{name}.pkl")
    joblib.dump(model, path)
    logger.info(f"Saved sklearn model {name} to {path}")

def load_sklearn_model(name: str, directory: str = "models"):
    path = os.path.join(directory, f"{name}.pkl")
    logger.info(f"Loading sklearn model from {path}")
    return joblib.load(path)

def save_lstm_model(model, name: str = "lstm_model", directory: str = "models"):
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{name}.h5")
    model.save(path)
    logger.info(f"Saved LSTM model to {path}")

def load_lstm_model(name: str = "lstm_model", directory: str = "models"):
    path = os.path.join(directory, f"{name}.h5")
    logger.info(f"Loading LSTM model from {path}")
    return tf.keras.models.load_model(path)
