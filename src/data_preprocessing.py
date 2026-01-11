import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path):
    df = pd.read_csv(path, header=None)
    return df


def split_data(df, test_size=0.2):
    X = df.drop(columns=[60], axis=1)
    y = df[60]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)

    return X_train, X_test, y_train, y_test
