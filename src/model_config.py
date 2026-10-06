import numpy as np
from sklearn.ensemble import  RandomForestClassifier
from sklearn.linear_model import  LogisticRegression
from sklearn.tree import  DecisionTreeClassifier
from src.config import RANDOM_STATE

LOGISTIC_REGRESSION_SEARCH_SPACE = {
    "model__C": np.arange(0.1, 10, 0.5),
}

DECISION_TREE_SEARCH_SPACE = {
    "model__max_depth": [3, 5, None],
    "model__min_samples_leaf": [2, 5],
}

RANDOM_FOREST_SEARCH_SPACE = {
    "model__max_depth": [5, None],
    "model__min_samples_leaf": [1, 3],
}



MODEL_CONFIGS = {

    "logistic_regression": {
        "estimator": LogisticRegression(
            max_iter = 2000,
            random_state  = RANDOM_STATE
        ),
        "search_space": LOGISTIC_REGRESSION_SEARCH_SPACE,
    },
    "decision_tree": {
        "estimator": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
        ),
        "search_space": DECISION_TREE_SEARCH_SPACE,
    },
    "random_forest": {
        "estimator": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "search_space": RANDOM_FOREST_SEARCH_SPACE,
    }
}