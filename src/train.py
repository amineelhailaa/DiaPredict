import os

import mlflow
import pandas as pd
from mlflow import MlflowClient
from sklearn.model_selection import train_test_split

from src import clustering_pipeline
from src.classification import create_model_pipeline
from src.config import RAW_DATA_PATH, FEATURES, RANDOM_STATE
from src.data import load_data, clean_data
from src.tuning import tune_model


def run_training():
    # Load the features in the same order for training and prediction.
    raw_data = load_data(RAW_DATA_PATH)
    cleaned_data = clean_data(raw_data)
    features = cleaned_data.loc[:, FEATURES]
    X_train, X_test = train_test_split(
        features,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    # Fit preprocessing and K-Means together on the training rows.
    cluster_pipeline = clustering_pipeline.Kmean_build_clustering_pipeline(
        n_clusters=3,
        imputer="knn",
    )
    cluster_pipeline.fit(X_train)
    y_train = pd.Series(
        cluster_pipeline.predict(X_train),
        index=X_train.index,
        name="Cluster",
    )

    y_test = pd.Series(
        cluster_pipeline.predict(X_test),
        index=X_test.index,
        name="Cluster",
    )

    profiles = clustering_pipeline.build_cluster_profiles(X_train, y_train)
    mapping = clustering_pipeline.build_cluster_mapping(profiles)


    search = tune_model("logistic_regression", X_train, y_train)
    class_model = search.best_estimator_
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment("DiaPredict")

    model_name = "DiaPredictClassifier"

    with mlflow.start_run(run_name="training-workflow"):
        model_info = mlflow.sklearn.log_model(
            sk_model= class_model,
            name="classifier",
            registered_model_name=model_name,
            metadata= {
                "features": FEATURES,
                "cluster_mapping": mapping
            }
        )

        MlflowClient().set_registered_model_alias(
            name=model_name,
            alias="champion",
            version=str(model_info.registered_model_version),
        )


    return {
        "model_version": model_info.registered_model_version,
        "mapping": mapping
    }