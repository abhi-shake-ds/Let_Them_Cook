"""Data loading, chronological split and feature/target separation."""

import pandas as pd

# Columns that must never be used as predictors:
#   instant, dteday -> record index and date (identifiers)
#   casual, registered -> cnt = casual + registered, so they leak the target
LEAK_AND_ID_COLS = ['instant', 'dteday', 'casual', 'registered']
TARGET = 'cnt'
SPLIT_DATE = '2012-07-01'


def load_hour_data(path='../data/raw/hour.csv'):
    """Load hour.csv sorted in time order (paths are relative to notebooks/)."""
    df = pd.read_csv(path, parse_dates=['dteday'])
    return df.sort_values(['dteday', 'hr']).reset_index(drop=True)


def chronological_split(df, cutoff=SPLIT_DATE):
    """Train = dates before `cutoff`, test = dates from `cutoff` onwards."""
    train = df[df['dteday'] < cutoff].copy()
    test = df[df['dteday'] >= cutoff].copy()
    return train, test


def get_xy(df):
    """Return the feature matrix (leak and id columns removed) and the target."""
    X = df.drop(columns=LEAK_AND_ID_COLS + [TARGET])
    return X, df[TARGET]
