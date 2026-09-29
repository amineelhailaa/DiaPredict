"""Service starter; build the clinical form in project task 22."""
import os

import requests
import streamlit as st

st.set_page_config(page_title="DiaPredict", page_icon="🧪")
st.title("DiaPredict")
st.info("Docker environment is ready. Train your model before implementing the prediction form.")
api_url = os.environ.get("API_URL", "http://fastapi:8000")
try:
    response = requests.get(f"{api_url}/health", timeout=5)
    response.raise_for_status()
    status = response.json()
    st.success("FastAPI is reachable.")
    st.json(status)
    if not status.get("model_ready"):
        st.warning("No prediction model is loaded yet.")
except (requests.RequestException, ValueError) as exc:
    st.error(f"FastAPI is unavailable: {exc}")
st.caption("Follow PROJECT_TASKS.md to implement clustering, classification, and inference.")
