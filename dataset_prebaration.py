from sklearn.model_selection import train_test_split
from math import ceil
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd

from lagged_features_extraction import calculate_lagged_features
from sklearn.ensemble import RandomForestRegressor


def extract_important_features(x_train, y):
    rf_model = RandomForestRegressor()
    rf_model.fit(x_train, y)
    feature_importances = pd.Series(rf_model.feature_importances_, index=x_train.columns)
    significant_features = feature_importances[feature_importances > 0.004].index

    return significant_features


def split(dataframe):
    y = dataframe["value"]
    feature_cols = [col for col in dataframe.columns if col != "value"]
    feature_cols = [col for col in feature_cols if col != "timestamp"]
    x = dataframe[feature_cols]
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, random_state=0, train_size=.8)

    return x_train, x_valid, y_train, y_valid


def create_features(dataframe):
    df = dataframe.copy()
    if "anomaly" in df.columns:
        df = df[df["anomaly"] != True]

        df.drop("anomaly", axis=1, inplace=True)

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour'] = df['timestamp'].dt.hour

    df['day_of_week'] = df['timestamp'].dt.day_of_week

    df['quarter'] = df['timestamp'].dt.quarter

    df['month'] = df['timestamp'].dt.month

    df['year'] = df['timestamp'].dt.year

    num_observations = len(df)
    period = int(num_observations / 2)

    decomposition = seasonal_decompose(df['value'], model='additive', period=period)

    df['trend'] = decomposition.seasonal.dropna()

    df = df.dropna()

    return df


def add_lagged_values(dataframe, lagged_values):

    for lag in range(1, lagged_values):
        dataframe[f'lag_{lag}'] = dataframe['value'].shift(lag)

    dataframe = dataframe.dropna()

    return dataframe
