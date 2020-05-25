import numpy as np
from sklearn.preprocessing import OneHotEncoder


class OneHotEncoderSupportingMissingValues():
    """
    Behaves exactly like sklearn's OneHotEncoder() except replaces missing values
    (represented by np.nan) with a np.nan vector.
    This implementation only allows a 1d numpy array.
    """

    def __init__(self, **params):
        self.onehotencoder = OneHotEncoder(**params)

    def fit(self, y):
        self.onehotencoder = self.onehotencoder.fit(
            y[~np.isnan(y)].reshape(-1, 1))
        return self

    def transform(self, y):
        part_enc_y = self.onehotencoder.transform(
            y[~np.isnan(y)].reshape(-1, 1))
        full_enc_y = np.empty((y.shape[0], part_enc_y.shape[1]))
        full_enc_y[:] = np.nan
        full_enc_y[~np.isnan(y).reshape(-1), :] = part_enc_y
        return full_enc_y

    def fit_transform(self, y):
        self.onehotencoder = self.onehotencoder.fit(
            y[~np.isnan(y)].reshape(-1, 1))
        part_enc_y = self.onehotencoder.transform(
            y[~np.isnan(y)].reshape(-1, 1))
        full_enc_y = np.empty((y.shape[0], part_enc_y.shape[1]))
        full_enc_y[:] = np.nan
        full_enc_y[~np.isnan(y).reshape(-1), :] = part_enc_y
        return full_enc_y

    def inverse_transform(self, enc_y):
        part_y = self.onehotencoder.inverse_transform(
            enc_y[~np.isnan(enc_y).any(axis=1)])
        full_y = np.empty((enc_y.shape[0], 1))
        full_y[:] = np.nan
        full_y[~np.isnan(enc_y).any(axis=1)] = part_y
        return full_y

    def get_feature_names(self, prefix):
        return self.onehotencoder.get_feature_names(prefix)
