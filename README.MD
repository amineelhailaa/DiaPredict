# DiaPredict

A learning project for K-Means patient grouping and classification of the resulting cluster labels. Follow [PROJECT_TASKS.md](PROJECT_TASKS.md) for the detailed implementation checklist.

## Start the Docker environment

Requirements: Docker Engine with Compose, and Python 3 on the host for the small configuration initializer.

```bash
python scripts/init_env.py
docker compose config --quiet
docker compose build
docker compose up -d --wait --wait-timeout 300
docker compose ps
docker compose exec jupyter python scripts/check_stack.py
```

The check script verifies network connectivity, dataset access, Airflow health, and MLflow artifact upload/download. It creates a clearly labeled `DiaPredict-infrastructure-check` experiment, not a trained model.

There are **six separate containers**:

| Service | Purpose | Address |
|---|---|---|
| Jupyter | EDA and model training | http://localhost:8888 |
| MLflow | Tracking, Registry, and artifact serving | http://localhost:5000 |
| FastAPI | API starter | http://localhost:8000/docs |
| Streamlit | UI starter | http://localhost:8501 |
| Airflow | Standalone orchestration | http://localhost:8080 |
| PostgreSQL | Airflow metadata only | Internal `airflow-db:5432` |

Airflow runs its own components together in one standalone container using LocalExecutor. No separate scheduler, worker, Redis, or initialization container is needed. PostgreSQL is separate. MLflow uses SQLite and artifact storage in its own persistent volume.

Get the Jupyter login token/URL and generated Airflow `admin` password:

```bash
docker compose exec jupyter jupyter server list
docker compose exec airflow cat /opt/airflow/state/passwords.json
```

Keep these local credentials private. `.env` is generated once, ignored by Git, and not overwritten by rerunning the initializer. All published ports bind only to localhost.

Jupyter follows DeliveryETA's simple Compose setup: its command is defined directly in `compose.yaml`, with no extra Jupyter configuration file or health check. Its local login token is `diapredict-local`; it does not use the generated `.env` Jupyter token. The `jupyter` service can be selected as a Docker Compose interpreter in PyCharm.

## Current implementation status

Dockerfiles, service-specific requirements, Compose health checks, and minimal web starters are supplied. **No model has been trained.** FastAPI `/health` is a liveness check; `/ready` and `/predict` return 503 until model loading and inference are implemented. Streamlit currently displays the API status. The training DAG is a later checklist task.

The CSV contains 768 rows, eight clinical features, and an exported row index. It has no diagnosis label. Classification will measure agreement with K-Means assignments, not diagnostic accuracy.

## Work inside the containers

The repository is mounted at `/workspace` in Jupyter and Airflow. Notebooks and generated models persist on the host. FastAPI and Streamlit have read-only application-code mounts.

```bash
docker compose exec jupyter bash
docker compose exec jupyter python -m pip check
docker compose logs --tail=100 airflow mlflow fastapi
# After editing API code:
docker compose restart fastapi
# After changing service dependencies:
docker compose up -d --build fastapi
# Stop without deleting named volumes:
docker compose down
```

`requirements.txt` describes the training environment. Shared model packages are pinned in `docker/requirements/model.txt`; the other requirement files contain service-specific packages. Airflow's training dependencies are installed into `/opt/airflow/training`, isolated from Airflow itself. Later DAGs should invoke `/opt/airflow/training/bin/python` as a subprocess with `/workspace` as the working directory. Jupyter and FastAPI do not contain Airflow.

Use `http://mlflow:5000` inside containers for MLflow and `http://fastapi:8000` for the API. Environment variables already provide those addresses. Browser URLs use localhost instead.

Named volumes retain Airflow's PostgreSQL data, Airflow logs/passwords, and MLflow metadata/artifacts. **`docker compose down -v` deletes these volumes**; do not use it for routine shutdown. Changing `.env` database credentials does not change passwords in an already initialized PostgreSQL volume.

MLflow's SQLAlchemy dependency is explicitly pinned because version 2.1 removes a connection-pool helper used by MLflow 3.6. Keep that compatibility pin when updating dependencies.

The Docker setup is for local development. See the checklist for model evaluation, Registry integration, the application, DAG implementation, and final delivery documentation.

## Verified setup

The five application images built successfully and all six containers passed their health checks. `scripts/check_stack.py` passed dataset access, service connectivity, Airflow health, and MLflow artifact upload/download. Jupyter's repository mount is writable, and Airflow's isolated training environment imports the model libraries.

The slim Jupyter image does not include the Git executable or a named system account for every possible host UID. MLflow may warn that automatic source-revision/user tags cannot be resolved. Artifact transfer still works; set the run's user/source-revision tags explicitly when implementing experiment tracking. No model training or clinical evaluation has been performed by these infrastructure checks.













