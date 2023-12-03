import mlflow
import os
import pandas as pd
import re
from sqlalchemy import create_engine

from lagged_features_extraction import calculate_lagged_features
from linear_regression_model import xgb_model_predictions
from dataset_prebaration import *


def run():
    path = "./train_splits"

    files = os.listdir(path)

    csv_files = [file for file in files if file.endswith(".csv")]

    for csv_file in csv_files:
        file_path = os.path.join(path, csv_file)
        historical_df = pd.read_csv(file_path, parse_dates=['timestamp'])

        time_difference_train = pd.to_datetime(historical_df['timestamp'].iloc[1]) - pd.to_datetime(
            historical_df['timestamp'].iloc[0])

        historical_df['value'] = historical_df['value'].fillna(historical_df['value'].mean())

        if pd.isna(historical_df['timestamp'].iloc[0]):
            historical_df['timestamp'].iloc[0] = (
                    pd.to_datetime(historical_df['timestamp'].bfill()).iloc[0] - time_difference_train
            )

        if historical_df['timestamp'].isna().any():
            nan_rows = historical_df['timestamp'].isna()
            historical_df.loc[nan_rows, 'timestamp'] = (
                    pd.to_datetime(historical_df['timestamp'].ffill()) + time_difference_train
            )

        historical_df.rename(columns={'timestamp': 'time'}, inplace=True)

        df = create_features(historical_df)

        number_of_lagged_features = calculate_lagged_features(historical_df['value'])

        df = add_lagged_values(df, number_of_lagged_features)

        df = df.ffill()

        X_train, X_valid, y_train, y_valid = split(df)

        selected_features = extract_important_features(X_train, y_train)

        significant_features_list = list(selected_features)

        params = {
            "lags": number_of_lagged_features,
            "significant_features": significant_features_list
        }

        X_train_selected = X_train[selected_features]

        mlflow.set_tracking_uri("sqlite:///mlflow.db")

        with mlflow.start_run():
            mlflow.log_params(params)

            match = re.search(r'\d+', csv_file)
            dataset_id = match.group() if match else None

            if dataset_id:
                mlflow.set_tag("dataset_id", dataset_id)

            model = xgb_model_predictions(X_train_selected, y_train)

            artifact_path = f"mlflow_model/{dataset_id}"
            registered_model_name = f"dataset_{dataset_id}"

            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=artifact_path,
                input_example=X_train_selected,
                registered_model_name=registered_model_name,
            )
