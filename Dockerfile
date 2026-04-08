FROM python:3.11-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV APP_HOST=0.0.0.0
ENV APP_PORT=8000
ENV ARTIFACTS_DIR=/app/artifacts

RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY static ./static
COPY templates ./templates
COPY examples ./examples
COPY scripts ./scripts
COPY artifacts ./artifacts
COPY README.md ./
COPY .env.example ./
COPY Makefile ./

RUN useradd --create-home --shell /bin/bash appuser \
    && mkdir -p /app/artifacts \
    && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --retries=5 CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health').read()"

CMD ["sh", "-c", "python scripts/bootstrap_runtime.py && uvicorn app.main:app --host ${APP_HOST} --port ${APP_PORT}"]
