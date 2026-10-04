from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from src.config import RANDOM_STATE

LOGISTIC_REGRESSION_SEARCH_SPACE = {
    "model__C": [0.1, 1.0, 10.0],
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
    "dummy_baseline": {
        "estimator": DummyClassifier(
            strategy="most_frequent",
        ),
        "search_space": None,
    },
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