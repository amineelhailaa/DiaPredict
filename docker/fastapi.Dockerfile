FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PYTHONPATH=/workspace
COPY docker/requirements/ /tmp/requirements/
RUN pip install -r /tmp/requirements/fastapi.txt && pip check
WORKDIR /workspace
COPY app/ ./app/
COPY src/ ./src/
EXPOSE 8000
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8000"]
