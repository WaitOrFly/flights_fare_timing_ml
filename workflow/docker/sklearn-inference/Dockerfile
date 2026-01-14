FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/sklearn-inference:1.2-1-cpu-py3

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

USER root
RUN mkdir -p /opt/ml/code \
    && chown -R sagemaker /opt/ml/code

ENV PYTHONUSERBASE=/opt/ml/code \
    PATH="/opt/ml/code/bin:${PATH}"

USER sagemaker
RUN python -m pip install --no-cache-dir --upgrade --user pip \
    && python -m pip install --no-cache-dir --user \
        scikit-learn==1.2.1 \
        numpy==1.26.4 \
        scipy==1.11.4 \
        pandas==2.2.2 \
        joblib==1.2.0
