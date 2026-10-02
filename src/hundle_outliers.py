import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class IQRWinsorizer(BaseEstimator, TransformerMixin):
    def __init__(self, multiplier=1.5):
        self.multiplier = multiplier

    def fit(self, X, y=None):
        self.feature_names_in_ = X.columns.to_numpy()
        X = np.asarray(X)
        q1 = np.nanquantile(X, 0.25, axis=0)
        q3 = np.nanquantile(X, 0.75, axis=0)

        iqr = q3 - q1

        self.lower_bounds_ = q1 - self.multiplier * iqr
        self.upper_bounds_ = q3 + self.multiplier * iqr

        return self

    def transform(self, X):
        X = np.asarray(X)
        return np.clip(X, self.lower_bounds_, self.upper_bounds_)


    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


