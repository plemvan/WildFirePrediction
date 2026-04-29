"""
Utility to load the Production model from the MLflow Model Registry.
"""

import logging
import os

import mlflow.pyfunc
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

REGISTERED_MODEL_NAME = "wildfire-xgboost-classifier"


def load_production_model():
    """
    Load the Production-stage model from the MLflow Model Registry.

    Returns
    -------
    mlflow.pyfunc.PyFuncModel
        The loaded Production model, ready for inference via model.predict(df).

    Raises
    ------
    mlflow.exceptions.MlflowException
        If the model or Production stage is not found in the registry.
    """
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "./mlruns")
    mlflow.set_tracking_uri(mlflow_uri)

    username = os.getenv("MLFLOW_TRACKING_USERNAME")
    password = os.getenv("MLFLOW_TRACKING_PASSWORD")
    if username:
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
    if password:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = password

    model_uri = f"models:/{REGISTERED_MODEL_NAME}/Production"
    logger.info("Loading model from registry: %s", model_uri)

    model = mlflow.pyfunc.load_model(model_uri)
    logger.info("Model loaded successfully.")
    return model


if __name__ == "__main__":
    model = load_production_model()
    logger.info("Registry load verification: OK — model type: %s", type(model))
