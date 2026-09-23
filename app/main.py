from fastapi import FastAPI
import logging
import joblib
import json
import pandas as pd
from pydantic import BaseModel


# --------------------------------------------------
# Logging Configuration
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s"
)

logger = logging.getLogger("opspredict")


# --------------------------------------------------
# FastAPI Application
# --------------------------------------------------

app = FastAPI(
    title="OpsPredict API",
    description="API for predicting late-delivery risk",
    version="1.0.0"
)


# --------------------------------------------------
# Load Trained Model
# --------------------------------------------------

model = joblib.load(
    "models/ops_predict_xgboost.pkl"
)

logger.info("OpsPredict XGBoost model loaded successfully")


# --------------------------------------------------
# Load Model Metadata
# --------------------------------------------------

with open("models/model_metadata.json", "r") as f:
    metadata = json.load(f)

FEATURES = metadata["features"]
THRESHOLD = metadata["threshold"]

logger.info(
    "Model metadata loaded successfully | Features: %d | Threshold: %.2f",
    len(FEATURES),
    THRESHOLD
)


# --------------------------------------------------
# Basic ML Monitoring Variables
# --------------------------------------------------

prediction_count = 0
late_prediction_count = 0
total_probability = 0.0


# --------------------------------------------------
# Prediction Input Schema
# --------------------------------------------------

class PredictionInput(BaseModel):
    same_state: int
    purchase_year: int
    purchase_month: int
    seller_previous_late_rate: float
    unique_sellers: int
    estimated_delivery_days: float
    total_items: int
    total_freight: float
    purchase_day: int
    seller_previous_late: int
    average_product_volume_cm3: float
    average_product_weight_g: float
    seller_previous_orders: int
    average_item_price: float
    total_product_volume_cm3: float


# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.get("/")
def root():

    logger.info("Health check requested")

    return {
        "message": "OpsPredict API is running"
    }


# --------------------------------------------------
# Prediction Endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(data: PredictionInput):

    global prediction_count
    global late_prediction_count
    global total_probability

    logger.info("Prediction request received")

    # Convert input into DataFrame
    input_data = pd.DataFrame(
        [data.model_dump()],
        columns=FEATURES
    )

    # Generate probability
    probability = float(
        model.predict_proba(input_data)[0][1]
    )

    # Apply decision threshold
    prediction = int(
        probability >= THRESHOLD
    )

    # Determine risk category
    if prediction == 1:
        risk = "Late"
    else:
        risk = "On Time / Early"

    # --------------------------------------------------
    # Update Monitoring Metrics
    # --------------------------------------------------

    prediction_count += 1
    total_probability += probability

    if prediction == 1:
        late_prediction_count += 1

    # Log prediction result
    logger.info(
        "Prediction completed | Risk: %s | Probability: %.4f | Threshold: %.2f",
        risk,
        probability,
        THRESHOLD
    )

    return {
        "prediction": prediction,
        "risk": risk,
        "late_delivery_probability": round(probability, 4),
        "threshold": THRESHOLD
    }


# --------------------------------------------------
# ML Monitoring Endpoint
# --------------------------------------------------

@app.get("/monitoring")
def monitoring():

    if prediction_count > 0:

        average_probability = (
            total_probability / prediction_count
        )

        late_prediction_percentage = (
            late_prediction_count / prediction_count
        ) * 100

    else:

        average_probability = 0.0
        late_prediction_percentage = 0.0

    logger.info(
        "Monitoring metrics requested | Predictions: %d | Late: %d",
        prediction_count,
        late_prediction_count
    )

    return {
        "total_predictions": prediction_count,
        "late_predictions": late_prediction_count,
        "late_prediction_percentage": round(
            late_prediction_percentage,
            2
        ),
        "average_late_probability": round(
            average_probability,
            4
        ),
        "decision_threshold": THRESHOLD
    }