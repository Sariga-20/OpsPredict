FROM python:3.13-slim

WORKDIR /app

COPY app/ ./app/
COPY models/ ./models/

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pandas \
    numpy \
    scikit-learn \
    xgboost \
    joblib

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]