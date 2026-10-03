from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.utils.validation import check_is_fitted
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer

from src.hundle_outliers import  IQRWinsorizer
from src.validation import (
    validate_age,
    validate_blood_pressure,
    validate_bmi,
    validate_diabetes_pedigree_function,
    validate_glucose,
    validate_insulin,
    validate_pregnancies,
    validate_skin_thickness,
)


VALIDATORS = {
    "Pregnancies": validate_pregnancies,
    "Glucose": validate_glucose,
    "BloodPressure": validate_blood_pressure,
    "SkinThickness": validate_skin_thickness,
    "Insulin": validate_insulin,
    "BMI": validate_bmi,
    "DiabetesPedigreeFunction": validate_diabetes_pedigree_function,
    "Age": validate_age,
}

SKEWED_FEATURES = [
    "BMI",
    "Insulin",
    "Pregnancies",
    "Age",
    "DiabetesPedigreeFunction",
]

NON_SKEWED_FEATURES = [
    "BloodPressure",
    "SkinThickness",
    "Glucose",
]


class InvalidValuesToNaN(BaseEstimator, TransformerMixin):


    def __init__(self, validators=None):
        if validators is None:
            validators = VALIDATORS
        self.validators = validators

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("InvalidValuesToNaN expects a pandas DataFrame")
        self.feature_names_in_ = X.columns.to_numpy()
        return self

    def transform(self, X):
        check_is_fitted(self, "feature_names_in_")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("InvalidValuesToNaN expects a pandas DataFrame")
        if not X.columns.equals(pd.Index(self.feature_names_in_)):
            raise ValueError(
                "Input columns must match the columns seen during fit, "
                "including their order"
            )
        transformed = X.copy()
        validators = self.validators if self.validators is not None else VALIDATORS
        for column, validator in validators.items():
            if column in transformed.columns:
                valid = transformed[column].map(validator)
                transformed.loc[~valid, column] = np.nan
        return transformed


    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return self.feature_names_in_
        return np.asarray(input_features, dtype=object)


def logarithm():
    return FunctionTransformer(np.log1p, feature_names_out="one-to-one")


def build_preprocessor(imputer="median"):
    if imputer == "knn":
        imputer_step = KNNImputer(
            n_neighbors=5,
            weights="distance",
        )
    elif imputer in {"mean", "median", "most_frequent"}:
        imputer_step = SimpleImputer(strategy=imputer)
    else:
        raise ValueError(
            "imputer must be 'knn', 'mean', 'median', or 'most_frequent'"
        )

    imputer_step.set_output(transform="pandas")

    skewed_pipeline = Pipeline(
        steps=[
            ("log", logarithm()),
            ("cap", IQRWinsorizer()),
        ]
    )

    outlier_transformer = ColumnTransformer(
        transformers=[
            ("skewed", skewed_pipeline, SKEWED_FEATURES),
            ("cap", IQRWinsorizer(), NON_SKEWED_FEATURES),
        ],
        verbose_feature_names_out=False,
    )

    return Pipeline(
        steps=[
            ("invalid_to_nan", InvalidValuesToNaN()),
            ("imputer", imputer_step),
            ("outliers", outlier_transformer),
            ("scaler", StandardScaler()),
        ]
    )
