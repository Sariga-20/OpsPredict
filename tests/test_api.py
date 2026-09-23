from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "OpsPredict API is running"


def test_predict():
    payload = {
        "same_state": 0,
        "purchase_year": 2018,
        "purchase_month": 8,
        "seller_previous_late_rate": 0.40,
        "unique_sellers": 3,
        "estimated_delivery_days": 8.0,
        "total_items": 5,
        "total_freight": 120.0,
        "purchase_day": 25,
        "seller_previous_late": 40,
        "average_product_volume_cm3": 5000.0,
        "average_product_weight_g": 2500.0,
        "seller_previous_orders": 100,
        "average_item_price": 150.0,
        "total_product_volume_cm3": 25000.0
    }

    response = client.post(
        "/predict",
        json=payload
    )

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert "risk" in result
    assert "late_delivery_probability" in result
    assert "threshold" in result

    assert result["prediction"] in [0, 1]
    assert 0 <= result["late_delivery_probability"] <= 1


def test_monitoring():
    response = client.get("/monitoring")

    assert response.status_code == 200

    result = response.json()

    assert "total_predictions" in result
    assert "late_predictions" in result
    assert "late_prediction_percentage" in result
    assert "average_late_probability" in result
    assert "decision_threshold" in result

    assert result["total_predictions"] >= 0
    assert result["late_predictions"] >= 0
    assert 0 <= result["late_prediction_percentage"] <= 100
    assert 0 <= result["average_late_probability"] <= 1
    assert 0 <= result["decision_threshold"] <= 1