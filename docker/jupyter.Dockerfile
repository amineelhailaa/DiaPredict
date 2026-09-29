FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /tmp/dependencies
COPY requirements.txt ./requirements.txt
COPY docker/requirements/ ./docker/requirements/
RUN pip install -r requirements.txt -r docker/requirements/jupyter.txt && pip check
WORKDIR /workspace
EXPOSE 8888
CMD ["python", "-m", "jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
