from sklearn.model_selection import train_test_split
from math import ceil
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd

from lagged_features_extraction import calculate_lagged_features
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest, mutual_info_regression

from timeseries_api.models import *


def extract_important_features(x_train, y):
    selector = SelectKBest(score_func=mutual_info_regression, k='all')
    x_train_selected = selector.fit_transform(x_train, y)
    selected_feature_indices = selector.get_support(indices=True)
    selected_features = x_train.columns[selected_feature_indices]

    for feature_name in selected_features:
        feature_entry = SelectedFeature(feature_name=feature_name)
        feature_entry.save()

    return selected_features


def split(dataframe):
    y = dataframe["value"]
    feature_cols = [col for col in dataframe.columns if col != "value"]
    feature_cols = [col for col in feature_cols if col != "time"]
    x = dataframe[feature_cols]
    x_train, x_valid, y_train, y_valid = train_test_split(x, y, random_state=0, train_size=.8)

    return x_train, x_valid, y_train, y_valid


def create_features(dataframe):
    df = dataframe.copy()
    if "anomaly" in df.columns:
        df = df[df["anomaly"] != True]

        df.drop("anomaly", axis=1, inplace=True)

    df['time'] = pd.to_datetime(df['time'])
    df['hour'] = df['time'].dt.hour

    df['day_of_week'] = df['time'].dt.day_of_week

    df['quarter'] = df['time'].dt.quarter

    df['month'] = df['time'].dt.month

    df['year'] = df['time'].dt.year

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
