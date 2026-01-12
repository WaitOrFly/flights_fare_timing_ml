FROM python:3.9-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    sagemaker==2.219.0 \
    xgboost==1.7.6 \
    scikit-learn==1.3.2

WORKDIR /opt/ml/processing
