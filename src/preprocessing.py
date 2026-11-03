import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

def _cyclic(X, period):
    X = np.asarray(X, dtype=float)
    return np.hstack([np.sin(2 * np.pi * X / period), np.cos(2 * np.pi * X / period)])

def _cyclic_names(transformer, input_features):
    return np.array([f'{c}_sin' for c in input_features] + [f'{c}_cos' for c in input_features])

def cyclic_transformer(period):
    return FunctionTransformer(_cyclic, kw_args={'period': period}, feature_names_out=_cyclic_names)

def build_preprocessor():
    return ColumnTransformer([
        ('hour', cyclic_transformer(24), ['hr']),
        ('month', cyclic_transformer(12), ['mnth']),
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['season', 'weathersit', 'weekday']),
        ('num', StandardScaler(), ['temp', 'atemp', 'hum', 'windspeed']),
        ('bin', 'passthrough', ['yr', 'holiday', 'workingday']),
    ])
