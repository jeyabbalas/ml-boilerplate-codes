from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
import numpy as np


class LabelEncoderSupportingMissingValues(TransformerMixin, BaseEstimator):

    def __init__(self):
        self.labelencoder = LabelEncoder()

    def fit(self, y):
        self.labelencoder = self.labelencoder.fit(y[~np.isnan(y)])
        return self

    def transform(self, y):
        enc_y = np.copy(y)
        enc_y[~np.isnan(enc_y)] = self.labelencoder.transform(y[~np.isnan(y)])
        return enc_y

    def fit_transform(self, y):
        self.labelencoder = self.labelencoder.fit(y[~np.isnan(y)])
        enc_y = np.copy(y)
        enc_y[~np.isnan(enc_y)] = self.labelencoder.transform(y[~np.isnan(y)])
        return enc_y

    def inverse_transform(self, y):
        inv_y = np.copy(y)
        inv_y[~np.isnan(inv_y)] = self.labelencoder.inverse_transform(
            y[~np.isnan(y)].astype(np.int))
        return inv_y
