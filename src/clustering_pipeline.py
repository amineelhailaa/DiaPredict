from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline

from src.config import RANDOM_STATE
from src.preprocessing import build_preprocessor


def Kmean_build_clustering_pipeline(
        n_clusters,
        imputer="knn",
        random_state= RANDOM_STATE
):
    return Pipeline(
        steps = [
            ("kmeans", KMeans(
                n_clusters=n_clusters,
                n_init=20,
                random_state=random_state
            ))
        ]
    )



