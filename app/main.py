from fastapi import FastAPI
import joblib
import json
import pandas as pd

app = FastAPI(
    title="OpsPredict API",
    description="API for predicting late-delivery risk",
    version="1.0.0"
)

# Load trained model
model = joblib.load(
    "models/ops_predict_xgboost.pkl"
)

# Load model metadata
with open("models/model_metadata.json", "r") as f:
    metadata = json.load(f)

FEATURES = metadata["features"]
THRESHOLD = metadata["threshold"]

from pydantic import BaseModel

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

@app.get("/")
def root():
    return {
        "message": "OpsPredict API is running"
    }
@app.post("/predict")
def predict(data: PredictionInput):

    input_data = pd.DataFrame(
        [data.model_dump()],
        columns=FEATURES
    )

    probability = float(
        model.predict_proba(input_data)[0][1]
    )

    prediction = int(
        probability >= THRESHOLD
    )

    if prediction == 1:
        risk = "Late"
    else:
        risk = "On Time / Early"

    return {
        "prediction": prediction,
        "risk": risk,
        "late_delivery_probability": round(probability, 4),
        "threshold": THRESHOLD
    }