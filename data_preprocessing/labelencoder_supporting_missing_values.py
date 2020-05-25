from sklearn.preprocessing import LabelEncoder
import numpy as np


class LabelEncoderSupportingMissingValues():
    """
    Behaves exactly like sklearn's LabelEncoder() except encodes missing values
    (represented by np.nan) with np.nan.
    """

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

    def inverse_transform(self, enc_y):
        y = np.copy(enc_y)
        y[~np.isnan(y)] = self.labelencoder.inverse_transform(
            enc_y[~np.isnan(enc_y)].astype(np.int))
        return y
