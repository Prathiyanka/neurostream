# Build context is the repo root (see docker-compose.yml) so the shared package is visible.
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY services/ingest/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
COPY services/common /opt/common
RUN pip install /opt/common
COPY services/ingest/ingest ./ingest
RUN useradd --create-home app
USER app
EXPOSE 8001
HEALTHCHECK --interval=10s --timeout=3s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8001/health').status==200 else 1)"
CMD ["python", "-m", "ingest.main"]
