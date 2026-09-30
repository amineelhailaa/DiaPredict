# DiaPredict — your step-by-step project checklist

Written on 29 September 2026 from your assignment and the actual project folder. Deadline in the brief: **9 October 2026 before midnight**.

This is an implementation guide, not a report of completed work. Tick each checkbox only after its “Done when” condition is satisfied. The Docker environment and minimal web-service starters now exist. Training code, prediction logic, and the training DAG remain implementation tasks. Code blocks for those later tasks are starting examples.

## 1. What you are building

Your project has two learning stages:

1. **Clustering:** K-Means groups patients with similar measurements without using a diagnosis label.
2. **Classification:** three classifiers learn to predict the cluster assigned by K-Means.

Then you save the winning pipeline, track experiments in MLflow, serve predictions through FastAPI and Streamlit, and automate retraining with Airflow and Docker Compose.

**Important:** your CSV has no `Outcome` or confirmed diabetes diagnosis column. Your target is `Cluster`, as requested in the brief. Classification scores measure agreement with K-Means labels, not accuracy at diagnosing diabetes. The risk mapping is the assignment's heuristic interpretation of cluster averages. Do not describe a classifier probability as a patient's probability of diabetes.

### A few terms before starting

| Term | Meaning in this project |
|---|---|
| Feature / `X` | Clinical measurements used as model inputs |
| Target / `y` | Cluster number the classifier learns to predict |
| `fit` | Learn values or model parameters from training data |
| `transform` | Apply an already fitted preprocessing operation |
| `predict` | Produce a cluster label for a patient |
| Hyperparameter | A setting you choose, such as tree depth or number of clusters |
| Validation / CV | Data splits used to compare settings during development |
| Test set | Reserved patients used for final evaluation |
| Pipeline | Preprocessing and model packaged into one reproducible object |
| Data leakage | Allowing evaluation data or the answer to influence training |
| Artifact | A saved output: model, plot, table, configuration, etc. |

## 2. What is actually in your folder

| Item | State observed |
|---|---|
| `data/raw/dataset-diabete-raw.csv` | 768 rows; 8 clinical columns plus an exported index |
| `notebooks/step1.ipynb` | One empty code cell |
| `requirements.txt` | Training dependencies; separate service requirements under `docker/requirements/` |
| `README.MD` | Docker quick start |
| `compose.yaml` | Six separate services, health checks, and persistent volumes |
| `src/`, `models/`, `dags/` | Training implementation still to complete |
| `app/` | Minimal API and Streamlit starters; no trained prediction model |
| `docker/` | Five separate Dockerfiles plus per-service requirements |

The clinical columns are `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, and `Age`. The first CSV column is unnamed and contains 0 through 767: it is an exported row index, not a clinical feature. There are no blank cells or duplicate clinical rows in the current file.

| Column | Zero values | Initial treatment to investigate/document |
|---|---:|---|
| Pregnancies | 111 | Keep zero; it is a valid count |
| Glucose | 5 | Treat zero as missing for this dataset |
| BloodPressure | 35 | Treat zero as missing |
| SkinThickness | 227 | Treat zero as missing |
| Insulin | 374 | Treat zero as missing; almost half the column |
| BMI | 11 | Treat zero as missing |
| DiabetesPedigreeFunction | 0 | No zero replacement needed |
| Age | 0 | No zero replacement needed |

376 rows contain at least one zero in the five columns above. Dropping all those rows would leave only 392 patients. Start with median imputation instead. Verify the dataset source and units before documenting them as facts; neither is established by the CSV alone.

## 3. Your immediate priority and schedule

Start with the supplied Docker environment, then work through the notebook and training tasks. Each application has its own container. Airflow runs in standalone mode inside one Airflow container, with a separate PostgreSQL database; there are no separate scheduler/worker/Redis containers.

**Today's target:** build and verify the six services in Task 01, then work through Tasks 02–09. Attempt Tasks 10–12 after preprocessing is clear. If time is very limited, reach a clean training table, documented missing-value decisions, and a reproducible train/test split. Do not spend the evening styling Streamlit.

| Date | Concrete target |
|---|---|
| 29 Sep | Separate Docker services, loading, split, EDA, preprocessing |
| 30 Sep | K-Means selection, cluster profiles, risk mapping |
| 1 Oct | Baseline and three classifiers |
| 2 Oct | Small hyperparameter searches, final evaluation, saved pipeline |
| 3 Oct | Extract notebook logic into reusable Python modules |
| 4 Oct | MLflow tracking and Registry |
| 5 Oct | FastAPI and Streamlit |
| 6 Oct | Validate trained-model integration across existing Docker services |
| 7 Oct | Airflow retraining workflow |
| 8 Oct | Integration checks, README, screenshots, Jira |
| 9 Oct | Clean startup rehearsal, presentation, final fixes and submission |

These are suggested work blocks, not measured estimates. If behind, reduce search grids and UI polish; keep every required deliverable represented.

## 4. Ordered implementation tasks

### Task 01 — Start the separate Docker services

The following infrastructure files are already supplied. No local Python virtual environment is needed; execute notebook/training code inside Jupyter's container.

| Compose service | Role | Browser address | Persistent storage |
|---|---|---|---|
| `jupyter` | Notebook editing, EDA, training | http://localhost:8888 | Repository mounted at `/workspace` |
| `mlflow` | Tracking server, Registry, artifact serving | http://localhost:5000 | `mlflow_data` volume: SQLite metadata and artifacts |
| `fastapi` | Prediction API; currently a starter | http://localhost:8000/docs | Code mounted read-only; future models loaded from Registry |
| `streamlit` | User interface; currently service status | http://localhost:8501 | Application code mounted read-only |
| `airflow` | All Airflow components in one standalone container | http://localhost:8080 | `airflow_state` volume: logs and generated login |
| `airflow-db` | PostgreSQL metadata database for Airflow only | Internal port 5432; no host port | `airflow_db` volume |

Airflow uses `LocalExecutor`; its internal scheduler, DAG processor, triggerer, and API server run together under `airflow standalone`. This is a local learning configuration. The MLflow metadata database is SQLite in its own persistent volume; it does not share Airflow's database.

- [ ] From the repository root, initialize your local environment file without overwriting an existing one:

```bash
python scripts/init_env.py
docker compose config --quiet
docker compose build
docker compose up -d --wait --wait-timeout 300
docker compose ps
```

`init_env.py` records your Linux user ID so notebook files remain editable on the host, and generates development credentials in ignored `.env`. It is safe to run again. `.env.example` documents the fields. Initial builds download Python packages and the Airflow image; allow time and disk space for them. The five Dockerfiles produce separate application images; PostgreSQL uses its official image.

- [ ] Open Jupyter. Obtain its tokenized login URL with:

```bash
docker compose exec jupyter jupyter server list
```

Use `http://localhost:8888` in your browser with the displayed token if the reported hostname is internal. Do not publish token URLs in screenshots.

- [ ] Log into Airflow as `admin`; retrieve the generated local password with:

```bash
docker compose exec airflow cat /opt/airflow/state/passwords.json
```

- [ ] Open MLflow, FastAPI `/docs`, and Streamlit. The API `/health` reports the process is alive, while `/ready` and `/predict` deliberately return HTTP 503 until you implement model loading/prediction. A healthy container does not yet mean a trained model exists.
- [ ] Verify dependencies and CSV access:

```bash
docker compose exec jupyter python -m pip check
docker compose exec jupyter python -c "import pandas as pd; print(pd.read_csv('data/raw/dataset-diabete-raw.csv').shape)"
docker compose exec airflow /opt/airflow/training/bin/python -m pip check
docker compose exec jupyter python scripts/check_stack.py
```

The check script creates a labeled `DiaPredict-infrastructure-check` MLflow run to verify artifact upload/download; it does not train a model. The raw CSV shape is `(768, 9)`, including the exported index. Task 03 removes that index.

- [ ] Understand the dependency split: `requirements.txt` holds training packages; `docker/requirements/model.txt` holds the shared model-runtime versions; each application has its own additional requirements. Airflow's own Python packages stay isolated from training packages in `/opt/airflow/training`. Future DAG tasks invoke that interpreter as a subprocess.
- [ ] Use container commands for ongoing work:

```bash
# Inspect problems without reinstalling packages blindly:
docker compose logs --tail=100 jupyter mlflow airflow
# Open a training shell:
docker compose exec jupyter bash
# Stop all services; named volumes are kept:
docker compose down
# Resume them:
docker compose up -d
```

Do not use `docker compose down -v` during normal work: it deletes the named database/artifact volumes. Notebooks and source files remain on your host through bind mounts.

- [ ] When dependencies change, edit the relevant requirements file and rebuild affected services. Keep model-library versions aligned between Jupyter, FastAPI, and Airflow's training environment. Top-level package versions are pinned; record `pip freeze` from each final image for the delivery rather than overwriting the shared requirements file with one service's environment.

**Done when:** all six containers are healthy, Jupyter can read the CSV, Airflow uses PostgreSQL, and the starter UI reaches FastAPI. Training and real prediction are later tasks.

### Task 02 — Organize the notebook and output folders

- [x] Keep `notebooks/step1.ipynb` as the main notebook. Add Markdown sections: objective, loading, split, EDA, preprocessing, K-Means, cluster interpretation, classification, tuning, evaluation, serialization, conclusions.
- [x] Create `data/processed/`, `reports/figures/`, and `reports/tables/`.
- [x] Define `RANDOM_STATE = 42`; use it consistently when adding randomized operations in later tasks.
- [x] Keep the raw CSV unchanged. Save derived datasets separately.

**Completed — 30 September 2026:** notebook outline is in place; shared seed and paths live in `src/config.py`, and `src/data.py` provides `load_data(path)`. The existing pandas/NumPy imports were preserved; output folders include `.gitkeep` files. Setup was checked and the raw CSV hash is unchanged. Modeling sections remain placeholders for the following tasks.

### Task 03 — Load and validate the data

- [ ] Load with pandas, inspect the columns, and remove only the verified exported index:

```python
from src.config import RAW_DATA_PATH
from src.data import load_data

df = load_data(RAW_DATA_PATH)
assert df["Unnamed: 0"].tolist() == list(range(len(df)))
df = df.drop(columns="Unnamed: 0")

FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]
assert df.columns.tolist() == FEATURES
assert df.shape == (768, 8)
display(df.head())
df.info()
display(df.describe().T)
print("Duplicate clinical rows:", df.duplicated().sum())
```

- [ ] Check numeric types, nonfinite values, negative measurements, and duplicate rows after removing the index.
- [ ] Keep the DataFrame index as a row identifier for split tracking, never as an input feature.
- [ ] Explain that `Pregnancies` is present in the file although omitted from the brief's introductory feature list. Start with all eight clinical inputs and document the choice.

**Done when:** you can explain every column and have an eight-column numeric table.

### Task 04 — Reserve the final test patients before learning preprocessing

The brief lists splitting after clustering. For a defensible held-out evaluation, reserve test rows first and fit the clustering workflow only on development rows. Document this deliberate ordering change.

- [ ] Create a fixed 80/20 split of the raw clinical rows:

```python
from sklearn.model_selection import train_test_split

X_train, X_test = train_test_split(
    df[FEATURES], test_size=0.20, random_state=42
)
# Expected with the current file: 614 development rows, 154 test rows.
assert set(X_train.index).isdisjoint(X_test.index)
```

- [ ] Save the row IDs to `data/processed/split_manifest.csv` with a `split` column.
- [ ] Do not use `stratify` here: cluster targets do not exist yet. Never create labels using all patients just to stratify this split.
- [ ] Use development rows for detailed EDA, feature decisions, clustering, and tuning. Keep test rows aside until the choices are frozen.

**Done when:** every row has a stable split, with no overlap. In this guide, `X_train` means the development set; cross-validation will split it further.

### Task 05 — Investigate missing values and distributions

- [ ] Report both `isna().sum()` and zero counts. Explain why no blank cells does not mean no missing measurements.
- [ ] On a development-data copy, replace zeros with `NaN` only in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI`.
- [ ] Compute missing percentages and save `reports/tables/missing_values.csv`.
- [ ] Draw histograms and boxplots of all eight columns. Add two or three sentences about skew, unusual values, and missingness.
- [ ] Preserve zeros in `Pregnancies`.

**Done when:** the notebook contains a missingness table and readable figures with written observations.

### Task 06 — Detect outliers and choose an explicit treatment

- [ ] Compute Q1, Q3 and IQR on the development data after zero replacement. Flag values below `Q1 - 1.5 * IQR` or above `Q3 + 1.5 * IQR`.
- [ ] Save counts by feature and count distinct affected rows; a row may be flagged in several columns.
- [ ] Inspect flagged examples. An IQR flag alone does not establish a data error.
- [ ] Start by retaining plausible extremes and using `StandardScaler`. Document why aggressive row deletion is costly with only 768 patients.
- [ ] If clipping is justified, implement a transformer that learns bounds in `fit` and applies them in `transform`, before scaling. Fit bounds inside each classifier training fold. Keep evaluation rows rather than silently dropping difficult cases.
- [ ] Record the policy per feature: retain, clip, or remove a verified invalid record, with counts and justification. If the evaluator expects actual removal, distinguish that requirement from statistical flagging and record the impact.

**Done when:** outliers were both detected and given a documented treatment decision. Do not automatically delete every high glucose or insulin observation.

### Task 07 — Explore relationships and select inputs

- [ ] Plot a correlation heatmap and a pairplot on development rows, with missing values handled for visualization.
- [ ] Compute variance, number of unique values, and missingness per feature.
- [ ] For the brief's variability requirement, also compare variance after training-fitted min-max scaling; unlike raw variance, this puts features on a common range. Inspect outlier sensitivity.
- [ ] Remove constant columns if any. Begin with all eight clinical features unless analysis provides a clear reason to exclude one.
- [ ] Document that raw variance depends on units, and that StandardScaler gives nonconstant features approximately unit variance—ranking variance after it is not useful.
- [ ] Consider an optional comparison excluding `Insulin` because of missingness, but do not add many experiments before the baseline works.
- [ ] Never include `Cluster`, `risk_category`, or the exported index in `X`.

**Done when:** a final ordered `FEATURES` list and a short keep/drop justification exist. Studying variability does not mean blindly retaining the three largest raw variances.

### Task 08 — Implement reusable preprocessing

- [ ] Create `src/__init__.py` and `src/preprocessing.py`.
- [ ] Implement an importable sklearn-compatible `ZeroToNaN` transformer using `BaseEstimator` and `TransformerMixin`. It should copy the input, replace zeros only in the specified columns, and preserve row order.
- [ ] Construct this preprocessing chain:

```text
Select ordered feature columns
→ ZeroToNaN
→ SimpleImputer(strategy="median")
→ optional learned clipping (only if justified)
→ StandardScaler()
```

- [ ] Use a `ColumnTransformer` or equivalent explicit selection step so input order is controlled.
- [ ] Confirm `fit_transform(X_train)` is finite and has the expected row count. For held-out or new data, use only `transform` or the complete pipeline's `predict`.
- [ ] Keep custom classes in `src/`, not just a notebook cell or lambda, so saved models can load in a fresh Python process.

**Done when:** one raw patient row, including permitted missing values or sentinel zeros, can be processed consistently.

### Task 09 — Choose the number of clusters

- [ ] Fit the preprocessing on development rows and transform them into `Z_train`.
- [ ] Try `k = 2, 3, 4, 5, 6` using `KMeans(n_clusters=k, n_init=20, random_state=42)`.
- [ ] For each value, record inertia, silhouette score, and cluster sizes.
- [ ] Plot inertia against `k` and silhouette against `k`.
- [ ] Select `k` using both plots, adequate group sizes, and interpretable profiles. Do not force `k=2` just because the final UI uses two colors.
- [ ] Explain: inertia measures within-cluster squared distances and decreases with more clusters; silhouette compares within-group cohesion with separation from other groups. A higher silhouette supports separation, not medical correctness.

**Done when:** `reports/tables/kmeans_selection.csv`, two figures, and a written choice of `k` exist. See the [official silhouette example](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis).

### Task 10 — Fit and freeze the clustering model

- [ ] Build a sklearn pipeline containing your raw-input preprocessing and the chosen K-Means estimator.
- [ ] Fit it on `X_train` only. Define `y_train = cluster_pipeline.predict(X_train)`.
- [ ] Save `models/clustering_pipeline.joblib`; also save `models/scaler.joblib` for the explicit scaler deliverable.
- [ ] Save the development rows with a `Cluster` column. Preserve original clinical units in exported tables.
- [ ] Once the whole model selection process is frozen, create `y_test = cluster_pipeline.predict(X_test)` with the same fitted clustering pipeline. Do not fit a second K-Means on test data.
- [ ] Eventually export combined train/test assignments with a `split` column, preserving the split manifest.

**Done when:** every development patient has a reproducible label, and the frozen pipeline can assign labels to unseen rows.

### Task 11 — Describe clusters and map the assignment's risk categories

- [ ] Group development patients by cluster and calculate counts, means, and medians in original units. Use the zero-cleaned measurements for descriptive summaries, report available-value counts, and make this choice explicit.
- [ ] Save `reports/tables/cluster_profiles.csv`.
- [ ] Apply the brief's rule to each cluster's means: `Glucose > 126 AND BMI > 30 AND DiabetesPedigreeFunction > 0.5`.
- [ ] Build a mapping from the actual cluster IDs. Never assume cluster `1` means high risk; IDs are arbitrary.
- [ ] Label qualifying clusters `high`; for a required binary demo, label the remaining clusters `low` **under this heuristic only**. Explain that failure to meet all three thresholds does not establish clinical safety.
- [ ] Handle zero qualifying clusters or multiple qualifying clusters honestly. Do not alter `k` or relabel a group merely to manufacture one red and one green badge.
- [ ] Save `models/risk_mapping.json`, including the thresholds, chosen `k`, and clustering artifact/run identifier. Add `risk_category` to the derived dataset.
- [ ] Plot cluster counts and a two-feature scatterplot. Optionally use a two-component PCA projection for visualization, explaining that clustering still used the selected clinical features.

**Done when:** every cluster ID has a traceable interpretation. Describe the thresholds as assignment rules, not independently validated diagnostic criteria.

### Task 12 — Understand exactly what classification evaluates

- [ ] Write this in your notebook: “The classifier approximates the assignments of a fixed K-Means model. Its target is a pseudo-label generated from the input measurements.”
- [ ] Use raw selected features as classifier `X`, and `Cluster` as `y`. Keep `risk_category` only for interpretation.
- [ ] Count the development labels and inspect imbalance.
- [ ] Explain why a classifier can achieve very high agreement: it is learning a partition already derived from the same feature space.
- [ ] Note that directly calling the clustering pipeline already assigns a cluster. The classifiers are included to satisfy the supervised-learning comparison and study how well each approximates that partition.

**Done when:** you can explain why high F1 is not evidence of high diabetes-detection accuracy.

### Task 13 — Set up shared cross-validation and a dummy baseline

- [ ] Use `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` on the development labels if every class has at least five rows. Otherwise reduce the folds appropriately; very tiny clusters need explicit discussion.
- [ ] Use the same fold assignments and input features for every classifier.
- [ ] Evaluate `DummyClassifier(strategy="most_frequent")` as a baseline.
- [ ] Choose **macro F1** as the primary comparison metric, with macro precision, macro recall, accuracy, and per-class support alongside it.
- [ ] Report mean and standard deviation over folds, not just the best fold.

**Evaluation limitation:** K-Means is frozen after fitting the entire development set, so its labels include the development validation rows' influence on the cluster geometry. These CV scores tune classifiers against that fixed labeler; they are not an unbiased evaluation of the entire discovery pipeline. The untouched outer test set provides the final agreement evaluation. A stricter end-to-end study would refit clustering inside each outer fold and handle cluster-ID alignment; that is beyond the minimum first implementation.

**Done when:** a baseline score and a shared CV configuration are recorded.

### Task 14 — Build three classification pipelines

- [ ] Train these three CPU-friendly models:

| Model | Starting configuration | What to understand |
|---|---|---|
| Logistic regression | `LogisticRegression(max_iter=2000, random_state=42)` | A simple linear decision boundary |
| Decision tree | `DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=42)` | Interpretable rules; can overfit |
| Random forest | `RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=1)` | Multiple trees combined |

- [ ] For each, use an `imblearn.pipeline.Pipeline` containing a fresh preprocessing object, an optional sampler, and the classifier.
- [ ] Start without resampling. Then compare `RandomOverSampler(random_state=42)` inside the pipeline to address the brief's imbalance requirement. It duplicates minority training examples; it does not create new independent evidence.
- [ ] Keep the sampler inside CV so only each training fold is resampled. Keep validation and test distributions unchanged.
- [ ] Prefer oversampling as the first experiment on this small dataset; undersampling discards scarce observations.

Pipeline structure:

```python
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from sklearn.linear_model import LogisticRegression

# build_preprocessor() is the function you implement in Task 08.
classifier_pipeline = Pipeline([
    ("preprocess", build_preprocessor()),
    ("sampler", "passthrough"),
    ("model", LogisticRegression(max_iter=2000, random_state=42)),
])
```

**Done when:** all three pipelines have comparable development CV results and you have assessed whether resampling helps. See [imbalanced-learn's leakage guidance](https://imbalanced-learn.org/stable/common_pitfalls.html).

### Task 15 — Tune small hyperparameter grids

- [ ] Use `GridSearchCV` on the full pipeline and only development data.
- [ ] Start with these small model grids; include `sampler: ["passthrough", RandomOverSampler(random_state=42)]` as an additional pipeline option:

| Model | Grid |
|---|---|
| Logistic regression | `model__C: [0.1, 1.0, 10.0]` |
| Decision tree | `model__max_depth: [3, 5, None]`; `model__min_samples_leaf: [2, 5]` |
| Random forest | `model__max_depth: [5, None]`; `model__min_samples_leaf: [1, 3]`; keep 200 trees |

- [ ] Use scoring names `f1_macro`, `precision_macro`, `recall_macro`, and `accuracy`, with `refit="f1_macro"`.
- [ ] Set the shared `cv`, use a modest `n_jobs` appropriate to your machine, and keep tree-estimator parallelism at one when the search already runs in parallel.
- [ ] Save `cv_results_`, `best_params_`, and the fitted `best_estimator_` for each model.
- [ ] Compare train and validation scores if investigating overfitting. Avoid huge grids or deep-learning detours.

**Done when:** each classifier has a selected setting and a searchable record of results.

### Task 16 — Freeze the winner and evaluate on the reserved test set

- [ ] Select the winner by development macro F1; for near ties, consider variability and simplicity. Record the decision before opening test results.
- [ ] Generate test pseudo-labels using the frozen clustering pipeline from Task 10.
- [ ] Evaluate each tuned classifier on the same test rows once, to meet the brief's comparison requirement. Do not change the chosen winner based on this table.
- [ ] Save per-model accuracy, macro precision, macro recall, macro F1, per-class classification reports, and confusion-matrix images.
- [ ] Specify the complete cluster label list when reporting. A class absent from the test set cannot have its recall meaningfully evaluated; show its support and explain the limitation.
- [ ] Explain precision as “how often predictions for this cluster are correct,” recall as “how many members of this cluster are recovered,” and F1 as a balance of the two. Macro averages give each cluster equal weight.
- [ ] Note the small test size (154 rows before any justified removals). Do not promise a particular score or claim precision beyond what the experiment supports.

**Done when:** `reports/tables/model_comparison.csv` and confusion matrices exist, with a short explanation of the selected model and limitations.

### Task 17 — Save and reload the final inference pipeline

- [ ] Save the fitted winning pipeline with `joblib.dump(best_pipeline, "models/classifier_pipeline.joblib")`.
- [ ] Save the ordered features, preprocessing policy, risk mapping, data hash, split identifiers, seed, Python/library versions, and teacher model reference.
- [ ] In a fresh process, load the artifact and predict on a small raw input DataFrame. Verify predictions exactly match those before serialization.
- [ ] Verify `src` custom classes are importable in the deployment environment.
- [ ] Keep the evaluated development-trained version as the initial deployed model. If you later refit on all rows, treat it as a new version and do not claim the old holdout score is a fresh independent evaluation of that refit.

**Done when:** inference requires one complete classifier pipeline, without manually scaling input beforehand. The standalone scaler is a deliverable, not an extra transformation to apply before pipeline prediction.

### Task 18 — Move reusable logic out of the notebook

- [ ] Keep the notebook for explanation and plots; move execution logic into these proposed modules:

```text
src/
  __init__.py
  preprocessing.py      # feature schema, zero handling, pipeline factory
  config.py             # shared constants and paths (already implemented)
  data.py               # CSV loader exists; extend with validation/split helpers
  clustering.py         # fit/select K-Means, profiles and risk mapping
  training.py           # classifier pipelines and search
  evaluation.py         # metrics and plots
  tracking.py           # MLflow logging and registration
  train.py              # command-line entry point
```

- [ ] Give functions explicit inputs and outputs; avoid notebook globals and absolute machine-specific paths.
- [ ] Later, add a small `config.json` for adjustable experiment settings (features, test fraction, candidate `k`, and search grids). Keep paths/default constants in `src/config.py`; if the seed becomes configurable, pass it explicitly instead of maintaining conflicting values.
- [ ] Implement and then verify a root-level command such as `docker compose exec jupyter python -m src.train --config config.json`.

**Done when:** a clean process can regenerate training outputs without manually executing notebook cells.

### Task 19 — Start MLflow tracking

- [ ] Use the MLflow service started in Task 01; do not launch a second server inside Jupyter.
- [ ] The environment sets `MLFLOW_TRACKING_URI=http://mlflow:5000` for Jupyter, FastAPI, and Airflow. Your browser uses `http://localhost:5000`. Create the `DiaPredict` experiment through the client.
- [ ] Metadata and artifacts persist under `/mlflow` in `mlflow_data`. Artifact serving lets clients upload/download through the server without sharing its filesystem.
- [ ] Log the clustering run: chosen `k`, seed, `n_init`, inertia, silhouette, scaler configuration, features, selection plots, cluster profiles, and the clustering/scaler artifacts.
- [ ] Log each classification run: estimator and sampler settings, CV results, final evaluation metrics, and confusion-matrix image. Use clear names such as `cv_f1_macro_mean` and `test_f1_macro`.
- [ ] Log the CSV SHA-256 hash, row count, split manifest, source revision when available, and dependency versions. Keep the data fingerprint separate from the MLflow model input/output signature.
- [ ] Log a raw input example and infer the model signature from representative raw inputs and predictions.

**Done when:** the UI shows traceable clustering and classification runs and downloadable artifacts. Integrate logging into reusable code so reruns are automatically tracked.

### Task 20 — Register and identify the validated version

- [ ] Register the complete inference pipeline under a name such as `DiaPredictClassifier`.
- [ ] Package the matching risk mapping and metadata with that version; do not keep an unrelated mutable mapping file in the web application.
- [ ] Verify loading by exact model version first.
- [ ] Assign a validated version an alias such as `champion`; the API can resolve `models:/DiaPredictClassifier@champion`.
- [ ] Address the brief's literal `Production` stage requirement: stages are deprecated in current MLflow documentation. Record this discrepancy in the README. If a literal stage is required, select and pin a version that supports it and verify its API; otherwise demonstrate an agreed alias-based promotion. Do not silently claim an alias is a stage.

**Done when:** the Registry points to the validated artifact and your promotion mechanism is explicit. See [MLflow Registry workflows](https://www.mlflow.org/docs/latest/ml/model-registry/workflow/).

### Task 21 — Implement the FastAPI service

- [ ] Extend the existing `app/api.py` starter into the real prediction service. Keep `/health` for liveness and make `/ready` return success only after the model is loaded.
- [ ] Define a Pydantic request model with the exact selected features. Specify which fields accept null for imputation, reject invalid types/nonfinite values, and enforce nonnegative values and integer counts where appropriate.
- [ ] Load the Registry pipeline and its matching metadata at startup. Cache the loaded model, expose its version, and fail readiness clearly if the model cannot be loaded.
- [ ] Convert a request into a one-row DataFrame using the saved feature order; call `pipeline.predict` directly on raw values.
- [ ] Map the predicted cluster to its versioned risk category. Return `cluster`, `risk_category`, `model_version`, and a short interpretation note.
- [ ] If showing a probability, label it cluster-assignment confidence, not diabetes probability; it is not automatically calibrated.
- [ ] Run `docker compose restart fastapi` after code changes and test through `http://localhost:8000/docs`. Rebuild with `docker compose up -d --build fastapi` if requirements change.

**Done when:** a valid request returns a structured prediction and malformed requests return a clear validation error.

### Task 22 — Implement the Streamlit interface

- [ ] Replace the service-status content in `app/streamlit_app.py` with one input per selected feature, readable labels, and units verified from dataset documentation.
- [ ] Allow an explicit “unknown” input where imputation is supported; do not force users to invent an insulin measurement.
- [ ] On submission, send the JSON request to FastAPI `/predict`; do not duplicate preprocessing in the UI.
- [ ] Show the returned category using the assignment's red/green badges, the model version, and a concise explanation that this is an experimental cluster-based category.
- [ ] For requested follow-up advice, use restrained wording such as “Discuss these measurements with a qualified clinician”; do not generate treatment instructions or imply a green badge rules out disease.
- [ ] Handle API timeouts, invalid input, and unavailable models visibly.
- [ ] Use `http://localhost:8501`; the existing container runs Streamlit. Its `API_URL` is `http://fastapi:8000`. Restart with `docker compose restart streamlit` when needed.

**Done when:** a form submission works end to end. The API loads from the Registry; Streamlit consumes that centrally managed model through the API.

### Task 23 — Verify the completed application across the existing containers

- [ ] Keep the six-service structure created in Task 01: Jupyter, MLflow, FastAPI, Streamlit, standalone Airflow, and Airflow PostgreSQL. No additional Airflow component containers are needed for this project setup.
- [ ] Check browser → Streamlit → FastAPI → MLflow Registry with a real trained model.
- [ ] Confirm the pipeline's custom transformers are importable from mounted `src/` and that artifact download works through MLflow.
- [ ] Keep internal addresses as `http://mlflow:5000` and `http://fastapi:8000`. The browser uses published localhost ports; PostgreSQL remains internal.
- [ ] Rebuild only the images whose requirements/Dockerfiles change. Source/notebook edits persist through bind mounts; restart FastAPI after source changes because automatic reload is not enabled.
- [ ] Verify database/model persistence by running `docker compose down`, then `docker compose up -d --wait --wait-timeout 300`. Never add `-v` for this check.
- [ ] Record final resolved dependencies per image and document any changed image versions.

**Done when:** all services are healthy and the real trained-model prediction path works across their network boundaries.

### Task 24 — Implement the Airflow DAG

- [ ] Use the supplied `apache/airflow:3.1.3-python3.12` image and its matching documentation. `airflow standalone` manages its components inside the single service.
- [ ] Keep the supplied standalone Compose configuration; the official multi-container quick start is a different topology. Refer to the [versioned standalone guide](https://airflow.apache.org/docs/apache-airflow/3.1.3/start.html) for this mode.
- [ ] Create `dags/diapredict_training.py` with this dependency chain:

```text
validate_data
→ prepare_split
→ fit_cluster_model_and_mapping
→ train_and_compare_classifiers
→ validate_candidate
→ register_candidate
→ promote_if_accepted
```

- [ ] Reuse the training CLI in a subprocess with `/opt/airflow/training/bin/python -m src.train --config /workspace/config.json`, using `/workspace` as its working directory. Import only Airflow/orchestration dependencies at DAG parse time; ML packages live in the isolated training environment. Do not import them directly into Airflow tasks or install them into Airflow's own environment. Never copy a second training implementation into the DAG.
- [ ] Use run-specific artifact paths. Pass small identifiers/paths between tasks, not full DataFrames or serialized models through XCom.
- [ ] Start with manual triggering and `catchup=False`. Add a schedule only once a full run succeeds.
- [ ] Set modest retries and useful task logs. A failed run must leave the previously deployed version available.
- [ ] Define promotion checks before running: valid schema, finite metrics, improvement over the dummy baseline on development validation, successful model reload, and complete mapping metadata. Choose any numeric acceptance threshold from documented project evidence, not an invented medical target.
- [ ] Keep the final held-out test set out of repeated promotion decisions. For retraining after new data arrives, define a new versioned validation protocol rather than repeatedly optimizing against the original test set.
- [ ] If K-Means is refitted, cluster IDs and their meanings can change. Recompute and version the mapping with the classifier; scores against different labelers are not automatically comparable. For direct old/new classifier comparisons, use the same frozen labeler.
- [ ] Document how the API picks up a promoted version, for example a controlled service restart that resolves the alias again.

**Done when:** one manual DAG run produces a candidate version with traceable artifacts; rejection does not replace the working model.

### Task 25 — Run focused integration checks

- [ ] Restart the notebook kernel and run every cell from top to bottom.
- [ ] Execute training from the repository root in a fresh process.
- [ ] Check feature columns exclude the label, risk category, and exported index.
- [ ] Check split IDs do not overlap and resampling occurs only during fitting.
- [ ] Compare predictions before and after serialization, then local and Registry-loaded predictions for identical inputs.
- [ ] Test one normal API request, one missing measurement, one invalid input, and an unavailable Registry/model situation.
- [ ] Check every possible classifier label has a mapping entry. Compare UI output to the API response.
- [ ] Restart containers and verify persisted Registry artifacts remain accessible.
- [ ] Trigger the DAG once successfully and test a rejected/failed candidate path that preserves the deployed model.

**Done when:** the main user flow and retraining flow work, with clear failure behavior. Focus tests on these real contracts rather than creating a large test suite for every plotting cell.

### Task 26 — Complete README and Jira documentation

- [ ] Fill the existing `README.MD` with project objective, dataset provenance, schema, missingness, split protocol, preprocessing decisions, selected `k`, cluster profiles, three-model comparison, actual metrics, and limitations.
- [ ] Include architecture, exact environment/setup commands, training command, service addresses, MLflow registration/promotion procedure, DAG trigger steps, and artifact locations.
- [ ] Explain that `Cluster` is the target and distinguish model agreement from clinical prediction.
- [ ] Add screenshots of cluster plots, model comparison, MLflow runs/Registry, the Streamlit result, and a successful Airflow run.
- [ ] Create Jira tickets from these numbered tasks. Suggested groups: data preparation, modeling, experiment tracking, application, orchestration, delivery. Add status, acceptance criteria, and evidence to each ticket.
- [ ] Put the actual Jira project/board link in the README and submission.

**Done when:** another person can understand and run the project from the README without asking you for undocumented steps.

### Task 27 — Rehearse the assessment and submit

- [ ] Prepare a ten-minute demonstration:
  - 1 minute: objective, data, and pseudo-label limitation.
  - 2 minutes: missing values, preprocessing, and K-Means choice.
  - 2 minutes: comparison of three classifiers and test results.
  - 2 minutes: live form → API → registered model prediction.
  - 2 minutes: MLflow evidence and successful Airflow run.
  - 1 minute: limitations and next steps.
- [ ] Practice explaining median imputation, standardization, silhouette, overfitting, precision/recall/F1, CV, leakage, pipelines, and the Registry.
- [ ] Be ready to explain why more duplicated training rows do not mean more independent patients, and why cluster number `1` has no fixed risk meaning.
- [ ] Check repository access, Jira access, required artifacts, and README commands before 9 October midnight. Keep screenshots as backup if a live service fails.

**Done when:** all assignment deliverables are present and you can explain every major design decision in your own words.

## 5. Proposed final repository layout

```text
DiaPredict/
├── README.MD
├── PROJECT_TASKS.md
├── requirements.txt
├── config.json
├── compose.yaml
├── .env.example
├── .gitignore
├── data/
│   ├── raw/dataset-diabete-raw.csv
│   └── processed/                # split IDs and derived cluster tables
├── notebooks/step1.ipynb
├── src/                          # reusable training/preprocessing code
├── models/                       # pipeline, scaler, clustering, metadata
├── reports/
│   ├── figures/
│   └── tables/
├── app/
│   ├── api.py
│   └── streamlit_app.py
├── dags/diapredict_training.py
├── docker/                       # five service Dockerfiles + per-service requirements
├── scripts/init_env.py           # initialize ignored local Compose configuration
└── tests/                        # focused data, inference, API checks
```

## 6. Common blockers and what to check first

| Symptom | First things to inspect |
|---|---|
| `FileNotFoundError` in notebook | Current directory and the root-path cell |
| K-Means complains about NaN | Imputer placement and columns entirely missing in the fitted subset |
| Unexpectedly excellent classification scores | Labels are derived from K-Means; also check target columns were excluded |
| Poor CV score but excellent training score | Tree depth, minimum leaf size, feature count, leakage, and rare clusters |
| Stratified CV fails | Smallest class count versus number of folds |
| Saved model cannot load custom transformer | Class defined in importable `src` module; same code/environment available |
| API result differs from notebook | Model version, feature order, double scaling, and matching mapping |
| Container cannot reach MLflow | Service DNS, server bind address, port, allowed-host configuration, health, artifact access |
| A retrained cluster changes risk color | Cluster IDs are arbitrary; version the new mapping with the new model |
| No cluster matches the high-risk rule | Report the actual result; do not fabricate a qualifying cluster |

## 7. Final deliverables checklist

- [ ] Complete notebook with comments, figures, results, and conclusions.
- [ ] Organized repository with reusable source modules.
- [ ] Working pinned requirements and recorded Python version.
- [ ] Saved classifier pipeline, clustering model, standalone scaler, and matching metadata.
- [ ] Three classifiers compared; imbalance handling and hyperparameter search documented.
- [ ] MLflow clustering/classification runs and registered validated pipeline.
- [ ] Explicit resolution of the deprecated `Production` stage requirement.
- [ ] FastAPI `/predict` serving the Registry model.
- [ ] Streamlit interface with risk display and appropriate interpretation.
- [ ] Dockerfiles and working Compose configuration.
- [ ] Airflow DAG demonstrated successfully.
- [ ] README, screenshots, actual metrics, limitations, and Jira link.
- [ ] Ten-minute demonstration rehearsed.

Technical references: [scikit-learn preprocessing and evaluation pitfalls](https://scikit-learn.org/stable/common_pitfalls.html), [silhouette example](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis), [imbalanced-learn resampling pitfalls](https://imbalanced-learn.org/stable/common_pitfalls.html), [MLflow Registry workflows](https://www.mlflow.org/docs/latest/ml/model-registry/workflow/), and [Airflow Docker setup](https://airflow.apache.org/docs/apache-airflow/stable/howto/docker-compose/index.html). Consult documentation matching your pinned versions when implementing commands and APIs.
