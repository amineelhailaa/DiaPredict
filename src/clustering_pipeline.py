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
            ("preprocessor", build_preprocessor(imputer)),
            ("kmeans", KMeans(
                n_clusters=n_clusters,
                n_init=20,
                random_state=random_state
            ))
        ]
    )


def build_cluster_profiles(X_train, y_train):
    # Keep the original measurements and attach each patient's cluster.
    profile_data = X_train.copy()
    profile_data["Cluster"] = y_train

    profiles = profile_data.groupby("Cluster").agg(
        patient_count=("Glucose", "size"),
        Glucose_mean=("Glucose", "mean"),
        BMI_mean=("BMI", "mean"),
        DiabetesPedigreeFunction_mean=("DiabetesPedigreeFunction", "mean"),
    )

    return profiles


def build_cluster_mapping(profiles):
    mapping = {}

    for cluster_id, profile in profiles.iterrows():
        is_high = (
            profile["Glucose_mean"] > 126
            and profile["BMI_mean"] > 30
            and profile["DiabetesPedigreeFunction_mean"] > 0.5
        )

        if is_high:
            risk_category = "high"
        else:
            risk_category = "low"

        mapping[str(cluster_id)] = {"risk_category": risk_category}

    return mapping
