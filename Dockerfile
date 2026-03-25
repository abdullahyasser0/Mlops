FROM python:3.10-slim

ARG RUN_ID

RUN pip install mlflow torch torchvision --no-cache-dir

WORKDIR /app

RUN echo "Downloading model for Run ID: ${RUN_ID}"

CMD ["python", "-c", "print('Model server ready')"]
