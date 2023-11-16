import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import pacf
from timeseries_api.models import *


def determine_lagged_features(file_path):
    df = pd.read_csv(file_path)
    df = df.ffill()

    lagged_features = calculate_lagged_features(df['value'])

    save_lagged_features_to_database(file_path, lagged_features)


def calculate_lagged_features(time_series):
    pacf_values = pacf(time_series)
    threshold = 5 / np.sqrt(len(time_series))
    significant_lags = np.where(np.abs(pacf_values) > threshold)[0]

    return len(significant_lags)


def save_lagged_features_to_database(file_path, lagged_features):
    dataset = TimeSeriesDatasets.objects.get(file=file_path)

    dataset.input_values = lagged_features
    dataset.save()
