import os
from datetime import datetime, timezone

from airflow.sdk import dag, task


@dag(
    dag_id="diapredict_etl",
    start_date=datetime(2026, 10, 8, tzinfo=timezone.utc),
    schedule=None,
    catchup=False,
    max_active_runs=1,
)
def etl():
    @task.external_python(
        python=os.environ.get(
            "TRAINING_PYTHON", "/opt/airflow/training/bin/python"
        ),
        expect_airflow=False,
        retries=0,
    )
    def train_champion():
        from src.train import run_training
        return run_training()
    train_champion()

etl_dag = etl()
