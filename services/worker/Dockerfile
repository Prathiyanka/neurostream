# Build context is the repo root (see docker-compose.yml) so the shared package is visible.
# No PyTorch here: the worker only needs ONNX Runtime, which keeps the image small.
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY services/worker/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
COPY services/common /opt/common
RUN pip install /opt/common
COPY services/worker/worker ./worker
COPY models /app/models
RUN useradd --create-home app
USER app
ENV MODEL_DIR=/app/models
CMD ["python", "-m", "worker.main"]
