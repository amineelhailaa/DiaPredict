FROM apache/airflow:3.1.3-python3.12
# Keep Airflow's own dependency environment unchanged. DAGs invoke this interpreter.
USER airflow
COPY --chown=airflow:root requirements.txt /tmp/training-dependencies/requirements.txt
COPY --chown=airflow:root docker/requirements/ /tmp/training-dependencies/docker/requirements/
RUN python -m venv /opt/airflow/training \
    && /opt/airflow/training/bin/pip install --no-cache-dir -r /tmp/training-dependencies/requirements.txt \
    && /opt/airflow/training/bin/pip check
RUN mkdir -p /opt/airflow/state && chmod 0775 /opt/airflow/state
CMD ["standalone"]
