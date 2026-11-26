FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app
COPY services/replayer/requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
COPY services/common /opt/common
RUN pip install /opt/common
COPY services/replayer/replayer ./replayer
RUN useradd --create-home app
USER app
ENTRYPOINT ["python", "-m", "replayer"]
CMD ["--help"]
