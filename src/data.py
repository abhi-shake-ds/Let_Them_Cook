import pandas as pd


LEAK_AND_ID_COLS = ['instant', 'dteday', 'casual', 'registered']
TARGET = 'cnt'
def load_hour_data(path='../data/raw/hour.csv'):
    df = pd.read_csv(path, parse_dates=['dteday'])
    return df.sort_values(['dteday', 'hr']).reset_index(drop=True)
def chronological_split(df, cutoff='2012-07-01'):
    train = df[df['dteday'] < cutoff].copy()
    test = df[df['dteday'] >= cutoff].copy()
    return train, test  
def get_xy(df):
    X = df.drop(columns=LEAK_AND_ID_COLS + [TARGET])
    return X, df[TARGET]
