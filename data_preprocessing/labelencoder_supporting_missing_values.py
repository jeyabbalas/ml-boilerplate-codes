from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
import numpy as np


class LabelEncoderSupportingMissingValues(TransformerMixin, BaseEstimator):

    def __init__(self):
        self.labelencoder = LabelEncoder()

    def fit(self, y):
        self.labelencoder = self.labelencoder.fit(y[y != None])
        return self

    def transform(self, y):
        enc_y = np.copy(y)
        enc_y[enc_y != None] = self.labelencoder.transform(y[y != None])
        return enc_y

    def fit_transform(self, y):
        self.labelencoder = self.labelencoder.fit(y[y != None])
        enc_y = np.copy(y)
        enc_y[enc_y != None] = self.labelencoder.transform(y[y != None])
        return enc_y

    def inverse_transform(self, y):
        inv_y = np.copy(y)
        inv_y[inv_y != None] = self.labelencoder.inverse_transform(
            y[y != None].astype(np.int))
        return inv_y
