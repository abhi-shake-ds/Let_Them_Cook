"""Preprocessing pipelines (fitted on training data only, inside a Pipeline).

Two variants are used because the model families need different inputs:

* ``kind='linear'``  - for Linear / Ridge / Lasso. A linear model cannot build
  interactions or two daily peaks by itself, so hour is one-hot encoded
  *per day type* (hour x working day = 48 columns). See 02-preprocessing.ipynb,
  section 1.6, for the cross-validation that chose this.
  ``atemp`` is dropped because it is almost identical to ``temp`` (r = 0.99).
* ``kind='forest'``  - for tree models. Trees find interactions on their own;
  in 04-evaluation.ipynb (section 4.7) one-hot hour made the forest worse, so
  cyclic hour and month are kept.
"""

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def _cyclic(X, period):
    X = np.asarray(X, dtype=float)
    return np.hstack([np.sin(2 * np.pi * X / period), np.cos(2 * np.pi * X / period)])


def _cyclic_names(transformer, input_features):
    return np.array([f'{c}_sin' for c in input_features] + [f'{c}_cos' for c in input_features])


def cyclic_transformer(period):
    return FunctionTransformer(_cyclic, kw_args={'period': period}, feature_names_out=_cyclic_names)


def _hour_daytype(X):
    """Combine hour and working-day flag into one label such as '8_1'."""
    X = np.asarray(X)
    hr = X[:, 0].astype(int).astype(str)
    wd = X[:, 1].astype(int).astype(str)
    return np.char.add(np.char.add(hr, '_'), wd).reshape(-1, 1)


def hour_daytype_encoder():
    """One-hot encode hour x working-day (24 x 2 = 48 columns)."""
    return Pipeline([
        ('combine', FunctionTransformer(_hour_daytype, feature_names_out=lambda t, f: np.array(['hr_wd']))),
        ('onehot', OneHotEncoder(handle_unknown='ignore')),
    ])


def build_preprocessor(kind='linear'):
    if kind == 'linear':
        return ColumnTransformer([
            ('hour_daytype', hour_daytype_encoder(), ['hr', 'workingday']),
            ('month', cyclic_transformer(12), ['mnth']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['season', 'weathersit', 'weekday']),
            ('num', StandardScaler(), ['temp', 'hum', 'windspeed']),
            ('bin', 'passthrough', ['yr', 'holiday']),
        ])
    if kind == 'forest':
        return ColumnTransformer([
            ('hour', cyclic_transformer(24), ['hr']),
            ('month', cyclic_transformer(12), ['mnth']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['season', 'weathersit', 'weekday']),
            ('num', StandardScaler(), ['temp', 'atemp', 'hum', 'windspeed']),
            ('bin', 'passthrough', ['yr', 'holiday', 'workingday']),
        ])
    raise ValueError("kind must be 'linear' or 'forest'")
