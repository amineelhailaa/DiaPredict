import pandas as pd
from sklearn.model_selection import train_test_split

from notebooks.step1 import y_train, y_test
from src import *
from src import clustering_pipeline
from src.config import RAW_DATA_PATH, FEATURES, RANDOM_STATE
from src.data import load_data, clean_data


def run_training():
    raw_data = load_data(RAW_DATA_PATH)
    cleaned_data = clean_data(raw_data)
    features = cleaned_data.loc[:, FEATURES]
    X_train, X_test = train_test_split(features, test_size=0.2, random_state=RANDOM_STATE)
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
        name="Cluster"
    )


