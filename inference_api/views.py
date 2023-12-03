from django.shortcuts import render
import pandas as pd
from rest_framework.views import APIView
from django.db import transaction
import json
import mlflow
from mlflow import MlflowClient
import ast
from django.http import JsonResponse

from dataset_prebaration import *
from linear_regression_model import *


class Mlflow(APIView):
    @transaction.atomic
    def post(self, request):
        body = request.body

        try:
            json_data = json.loads(body)

            id = json_data['dataset_id']

        except json.JSONDecodeError as e:
            return HttpResponse(f'Error decoding JSON: {str(e)}', status=400)

        path = json_data['test_dataset_path']

        historical_df = pd.read_csv(path, parse_dates=['timestamp'])

        time_difference_train = pd.to_datetime(historical_df['timestamp'].iloc[1]) - pd.to_datetime(
            historical_df['timestamp'].iloc[0])

        mlflow.set_tracking_uri("sqlite:///mlflow.db")

        run = mlflow.search_runs(filter_string=f"tags.dataset_id = '{id}'").iloc[0]

        run_id = run['run_id']
        lags = run["params.lags"]

        significant_features = run["params.significant_features"]

        df = pd.read_csv(path, parse_dates=['timestamp'])

        target_timestamp = json_data['start_timestamp']

        target_index = df[df['timestamp'] == target_timestamp].index[0]

        start_index = max(0, target_index - int(lags))

        rows_before_target = df.iloc[start_index:target_index]

        rows_before_target['value'] = rows_before_target['value'].fillna(rows_before_target['value'].mean())
        rows_before_target['timestamp'] = rows_before_target['timestamp'].ffill()

        rows_before_target = rows_before_target.loc[:, ['timestamp', 'value']]

        last_timestamp = rows_before_target['timestamp'].iloc[-1]

        last_timestamp_datetime = pd.to_datetime(last_timestamp)

        next_timestamp = last_timestamp_datetime + time_difference_train

        new_row = pd.DataFrame({'time': [next_timestamp], 'value': [None]})

        rows_before_target = pd.concat([rows_before_target, new_row], ignore_index=True)

        rows_before_target = rows_before_target.drop("time", axis=1)

        rows_before_target.rename(columns={'timestamp': 'time'}, inplace=True)

        rows_before_target['value'] = rows_before_target['value'].fillna(rows_before_target['value'].mean())

        X_test = create_features(rows_before_target)

        X_test = add_lagged_values(X_test, int(lags))

        feature_cols = [col for col in X_test.columns if col != "value"]
        feature_cols = [col for col in feature_cols if col != "time"]
        X_test = X_test[feature_cols]

        significant_features_list = ast.literal_eval(significant_features)

        X_test_selected = X_test[significant_features_list]

        artifact_path = f"mlruns/0/{run_id}/artifacts/mlflow_model/{id}"
        loaded_model = mlflow.sklearn.load_model(artifact_path)
        predictions = loaded_model.predict(X_test_selected)

        predicted_next_value = predictions[-1].item()

        return JsonResponse({'prediction': predicted_next_value, 'timestamp': target_timestamp})
