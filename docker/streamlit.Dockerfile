FROM python:3.12-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
COPY docker/requirements/streamlit.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && pip check
WORKDIR /workspace
COPY app/ ./app/
EXPOSE 8501
CMD ["streamlit", "run", "app/streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--browser.gatherUsageStats=false"]
