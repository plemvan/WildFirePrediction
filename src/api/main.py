"""
Wildfire Prediction API
Exposes the production XGBoost model via FastAPI.
"""

import logging
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data.df_aggregated import FEATURES
from src.models.registry_loader import load_production_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Model loaded once at startup via lifespan
# ---------------------------------------------------------------------------
model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    logger.info("Loading model from MLflow Registry...")
    model = load_production_model()
    logger.info("Model loaded successfully.")
    yield
    model = None


app = FastAPI(
    title="Wildfire Prediction API",
    description="Predicts wildfire probability from meteorological and environmental features.",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------
class WildfireFeatures(BaseModel):
    pr: float = Field(..., description="Precipitation (mm)")
    rmax: float = Field(..., description="Maximum relative humidity (%)")
    rmin: float = Field(..., description="Minimum relative humidity (%)")
    sph: float = Field(..., description="Specific humidity (kg/kg)")
    srad: float = Field(..., description="Surface downward shortwave radiation (W/m²)")
    tmmn: float = Field(..., description="Minimum temperature (K)")
    tmmx: float = Field(..., description="Maximum temperature (K)")
    vs: float = Field(..., description="Wind speed at 10m (m/s)")
    vpd: float = Field(..., description="Vapor pressure deficit (kPa)")
    fm100: float = Field(..., description="100-hour dead fuel moisture (%)")
    fm1000: float = Field(..., description="1000-hour dead fuel moisture (%)")
    erc: float = Field(..., description="Energy release component")
    bi: float = Field(..., description="Burning index")
    etr: float = Field(..., description="Reference evapotranspiration (mm)")
    pet: float = Field(..., description="Potential evapotranspiration (mm)")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "pr": 0.0,
                    "rmax": 45.0,
                    "rmin": 12.0,
                    "sph": 0.004,
                    "srad": 320.0,
                    "tmmn": 285.0,
                    "tmmx": 305.0,
                    "vs": 5.2,
                    "vpd": 1.8,
                    "fm100": 8.5,
                    "fm1000": 12.0,
                    "erc": 65.0,
                    "bi": 55.0,
                    "etr": 7.5,
                    "pet": 6.0,
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    wildfire: int = Field(..., description="Predicted class: 1 = wildfire, 0 = no wildfire")
    probability: float = Field(..., description="Predicted probability of wildfire")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Monitoring"])
def health():
    """Liveness check — returns 200 if the API is running."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(features: WildfireFeatures):
    """
    Predict wildfire probability from meteorological features.

    The model is logged via mlflow.sklearn, so the underlying
    XGBoostClassifierWrapper exposes predict_proba() directly.
    We unwrap it from the pyfunc flavour to access probabilities.
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    data = pd.DataFrame([features.model_dump()])[FEATURES]

    # model is a mlflow.pyfunc.PyFuncModel wrapping an sklearn model.
    # .predict() on pyfunc returns classes (0/1), not probabilities.
    # We unwrap the sklearn estimator to access predict_proba() directly.
    try:
        sklearn_model = model.unwrap_python_model()
    except Exception:
        # Standard sklearn flavour: the estimator lives here
        sklearn_model = model._model_impl.sklearn_model

    proba = float(sklearn_model.predict_proba(data.to_numpy())[0, 1])
    label = int(proba >= 0.5)

    # Log inputs and outputs for monitoring
    logger.info(
        "PREDICT | inputs=%s | wildfire=%d | probability=%.4f",
        features.model_dump(),
        label,
        proba,
    )

    return PredictionResponse(
        wildfire=label,
        probability=round(proba, 4),
    )
