# OpsPredict — Predictive Business Operations AI

OpsPredict is a machine learning system that predicts the risk of late e-commerce deliveries before the delivery happens.

The project uses historical order, customer, product, payment, and seller information to engineer predictive features and train an XGBoost classification model.

---

## 🎯 Business Problem

Late deliveries can negatively affect customer satisfaction and operational performance.

OpsPredict analyzes order-level information to estimate the probability that an order will be delivered late. The prediction can help identify higher-risk orders before delivery and support proactive operational decision-making.

---

## 🏗️ Architecture

```text
E-commerce Data
       ↓
Data Cleaning & Feature Engineering
       ↓
Time-Based Train/Test Split
       ↓
XGBoost Model
       ↓
SHAP Explainability
       ↓
MLflow Experiment Tracking
       ↓
FastAPI REST API
       ↓
Docker Container
       ↓
GitHub Actions CI
```
---

## 📊 Dataset

The project uses the Olist Brazilian E-Commerce Public Dataset.

The final modeling dataset contains **96,470 delivered orders** with features related to:

- Order timing
- Delivery estimates
- Order items
- Product characteristics
- Freight
- Payments
- Seller history
- Geographic information

---

### Target Variable

`delivered_late`

- `0` = On Time / Early
- `1` = Late

The dataset contains:

- **88,644** on-time/early orders
- **7,826** late orders
- Late-delivery rate: **8.11%**

---

## 🤖 Machine Learning

Several classification models were evaluated during experimentation:

- Logistic Regression
- Random Forest
- XGBoost

A chronological train/test split was used to avoid randomly mixing historical and future orders.

The final model uses **XGBoost** with the following configuration:

| Parameter | Value |
|---|---:|
| n_estimators | 300 |
| max_depth | 3 |
| learning_rate | 0.03 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| scale_pos_weight | 5 |
| Decision threshold | 0.70 |

The final model uses **15 selected features**, including delivery estimates, seller historical performance, freight, order characteristics, and geographic information.

---

## 📈 Model Evaluation

The model was evaluated using ROC-AUC and PR-AUC because the target variable is imbalanced.

### Validation Performance

The validation set was used for model selection and threshold tuning.

- ROC-AUC: **0.6793**
- PR-AUC: **0.1735**
- Selected decision threshold: **0.70**

### Final Chronological Test Performance

The final test set represents a later time period that was kept untouched during model selection.

- ROC-AUC: **0.5937**
- PR-AUC: **0.0709**

The lower performance on the later test period indicates **temporal distribution shift** between the validation period and the final test period.

No test-set threshold tuning was performed.

---

## 🔍 SHAP Explainability

SHAP (SHapley Additive exPlanations) was used to understand which features contribute most to the model's delivery-risk predictions.

The main features identified by the SHAP analysis include:

1. `estimated_delivery_days`
2. `purchase_month`
3. `same_state`
4. `total_freight`
5. `seller_previous_late_rate`
6. `purchase_year`
7. `seller_previous_late`
8. `total_items`
9. `purchase_day`
10. `seller_previous_orders`

SHAP summary plots are included in the `reports/` directory to provide both global feature importance and feature-level contribution analysis.

## 🏪 Seller Historical Features

OpsPredict includes historical seller-performance features calculated using only information available from previous orders.

The engineered features are:

- `seller_previous_orders`
- `seller_previous_late`
- `seller_previous_late_rate`

These features help the model incorporate historical seller delivery performance while avoiding the use of future order information during feature engineering.

## 🚀 FastAPI

The trained XGBoost model is exposed through a FastAPI REST API.

### Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Check whether the API is running |
| POST | `/predict` | Predict late-delivery risk |

The `/predict` endpoint accepts the model's 15 required features and returns:

- Prediction
- Risk classification
- Late-delivery probability
- Decision threshold

Interactive API documentation is available through FastAPI Swagger UI at:

```text
http://127.0.0.1:8000/docs
```

## 🐳 Docker

OpsPredict can be packaged and run as a Docker container.

### Build the Docker Image

```bash
docker build -t opspredict-api .
```

### Run the Docker Container

```bash
docker run -d -p 8000:8000 --name opspredict-container opspredict-api
```
### The API can then be accessed at:

```text
http://127.0.0.1:8000
```
### Swagger documentation:

```text
http://127.0.0.1:8000/docs
```
---

## ☁️ Live Deployment

OpsPredict is deployed as a Dockerized FastAPI service on Render.

### Public API

[Open OpsPredict API](https://opspredict.onrender.com)

### Swagger API Documentation

[Open Swagger Documentation](https://opspredict.onrender.com/docs)

The deployed API provides:

- API health check
- `/predict` endpoint
- Late-delivery probability prediction
- Risk classification based on the selected threshold
---

## 🧪 Testing & CI

Pytest is used to test the FastAPI application.

The project includes tests for:

- API health check
- `/predict` endpoint
- Prediction response structure
- Prediction probability range

Tests can be run locally with:

```bash
pytest tests/test_api.py -v
```

### CI Pipeline

```text
Push / Pull Request
        ↓
GitHub Actions
        ↓
Install Python & Dependencies
        ↓
Run Pytest
        ↓
Pass / Fail
```

---

## 📁 Project Structure

```text
OpsPredict/
│
├── app/
│   └── main.py
│
├── data/
│   ├── processed/
│   └── raw/
│
├── models/
│   ├── ops_predict_xgboost.pkl
│   └── model_metadata.json
│
├── notebooks/
│   ├── 01_data_exploration_and_feature_engineering.ipynb
│   └── 02_model_experimentation.ipynb
│
├── reports/
│   ├── shap_feature_importance.png
│   └── shap_summary_plot.png
│
├── src/
│
├── tests/
│   └── test_api.py
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── Dockerfile
├── requirements.txt
├── pytest.ini
├── .gitignore
└── README.md

```
## 🛠️ Technology Stack

### Programming & Data

- Python
- Pandas
- NumPy
- Scikit-learn

### Machine Learning

- XGBoost
- SHAP
- Logistic Regression
- Random Forest

### API & Deployment

- FastAPI
- Uvicorn
- Docker

### MLOps & Testing

- MLflow
- Pytest
- GitHub Actions

### Development Tools

- Jupyter Notebook
- VS Code
- Git
- GitHub
--- 
## 🔮 Future Improvements

Potential future improvements for OpsPredict include:

- Deploy the FastAPI service to a cloud platform
- Add automated model retraining
- Implement production monitoring and data-drift detection
- Improve late-delivery prediction using additional temporal and operational features
- Add a Streamlit interface for interactive predictions
- Integrate real-time order data
- Expand model explainability and operational recommendations
---

## 👩‍💻 Author

**Sariga C**

Computer Science Engineering Graduate | Data Analytics | Machine Learning | AI

### Project Repository

[GitHub Repository](https://github.com/Sariga-20/OpsPredict)