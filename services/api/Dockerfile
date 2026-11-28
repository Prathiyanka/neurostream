# Build context is the repo root (see docker-compose.yml) so the shared package is visible.
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY services/api/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
COPY services/common /opt/common
RUN pip install /opt/common
COPY services/api/api ./api
COPY services/api/migrations ./migrations
COPY services/api/alembic.ini ./alembic.ini
# Model files are only read to list models and verify hashes before a hot-swap.
COPY models /app/models
RUN useradd --create-home app
USER app
ENV MODEL_DIR=/app/models
EXPOSE 8000
HEALTHCHECK --interval=10s --timeout=3s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"
CMD ["python", "-m", "api.main"]
