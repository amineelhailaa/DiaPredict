from sklearn import clone
from sklearn.model_selection import KFold, cross_validate, StratifiedKFold
from imblearn.pipeline import Pipeline
from src.config import RANDOM_STATE
from src.model_config import MODEL_CONFIGS


def create_model_pipeline(model):
    return Pipeline(
        [   ("sampler", "passthrough"),
            ("model", clone(model)),
        ]
    )


def create_pipelines():
    return {
        model_name: create_model_pipeline(model["estimator"])
        for model_name, model in MODEL_CONFIGS.items()
    }


def train_pipeline(pipeline, X_train, y_train):
    return pipeline.fit(X_train, y_train)


def train_pipelines(X_train, y_train):
    return {
        model_name: train_pipeline(pipeline, X_train, y_train)
        for model_name, pipeline in create_pipelines().items()
    }



def create_cross_validation(k=5):
    return StratifiedKFold(
        n_splits=k,
        shuffle=True,
        random_state=RANDOM_STATE
    )


def cross_validate_pipeline(pipeline, X_train, y_train, cv):
    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring={
            "accuracy": "accuracy",
            "precision": "precision_macro",
            "recall": "recall_macro",
            "f1": "f1_macro",
        },
        return_train_score=True
    )
    return {
        "validation_accuracy_mean": scores["test_accuracy"].mean(),
        "validation_accuracy_std": scores["test_accuracy"].std(),
        "validation_precision_mean": scores["test_precision"].mean(),
        "validation_recall_mean": scores["test_recall"].mean(),
        "validation_recall_std": scores["test_recall"].std(),
        "validation_f1_mean": scores["test_f1"].mean(),
        "validation_f1_std": scores["test_f1"].std(),
        "training_accuracy_mean": scores["train_accuracy"].mean(),
        "training_accuracy_std": scores["train_accuracy"].std(),
        "training_precision_mean": scores["train_precision"].mean(),
        "training_precision_std": scores["train_precision"].std(),
        "training_recall_mean": scores["train_recall"].mean(),
        "training_recall_std": scores["train_recall"].std(),
        "training_f1_mean": scores["train_f1"].mean(),
        "training_f1_std": scores["train_f1"].std(),
    }


def cross_validate_pipelines(X_train, y_train):
    pipelines = create_pipelines()
    cv = create_cross_validation()
    results = {}

    for model_name, pipeline in pipelines.items():
        model_results = cross_validate_pipeline(
            pipeline,
            X_train,
            y_train,
            cv
        )
        results[model_name] = model_results
    return results
