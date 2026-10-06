from imblearn.over_sampling import SMOTE
from imblearn import pipeline
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from src.model_config import *
from src.classification import create_model_pipeline, create_cross_validation


def tune_model(model_name, X_train, y_train):
    config = MODEL_CONFIGS[model_name]
    estimator = config["estimator"]
    search_space = config["search_space"]

    if search_space is None:
        raise ValueError( f"{model_name} does not have a tuning serach space")


    #applying smote
    search_space = {
        **search_space,
        "sampler": ["passthrough", SMOTE()]
    }

    pipeline = create_model_pipeline(estimator)
    tuning_cv = create_cross_validation(5)

    search = GridSearchCV(
        estimator = pipeline,
        param_grid= search_space,
        scoring={
            "accuracy": "accuracy" ,
            "precision": "precision_macro",
            "recall" : "recall_macro",
            "f1": "f1_macro"
        },
        cv = tuning_cv,
        refit="f1",
        error_score="raise"
    )

    search.fit(X_train, y_train)
    return search