import pandas as pd
from rest_framework.views import APIView
from rest_framework.generics import ListCreateAPIView
from rest_framework.pagination import PageNumberPagination
from django.db import transaction
from django.http import HttpResponse
import json
from django.http import JsonResponse

from .serializers import *
from upload_datasets import upload_all_datasets
from dataset_prebaration import *
from linear_regression_model import *


class ListDatasets(ListCreateAPIView):
    queryset = TimeSeriesDatasets.objects.all()
    serializer_class = TimeSeriesSerializer
    pagination_class = PageNumberPagination


class UploadDatasets(APIView):
    @transaction.atomic
    def post(self, request):
        body = request.body

        try:
            json_data = json.loads(body)
            path = json_data["path"]

        except json.JSONDecodeError as e:
            return HttpResponse(f'Error decoding JSON: {str(e)}', status=400)

        upload_all_datasets(path)
        return HttpResponse('POST request processed successfully')


class ForecastPrediction(APIView):
    @transaction.atomic
    def post(self, request):
        body = request.body

        try:
            json_data = json.loads(body)

            id = json_data['dataset_id']
            lagged_values = json_data["values"]

        except json.JSONDecodeError as e:
            return HttpResponse(f'Error decoding JSON: {str(e)}', status=400)

        try:
            dataset = TimeSeriesDatasets.objects.get(dataset_id=id)

        except TimeSeriesDatasets.DoesNotExist:
            return JsonResponse({'error': 'Dataset not found'}, status=404)

        if len(lagged_values) != dataset.input_values:
            return JsonResponse({'Please, enter correct number of lagged values ': dataset.input_values})

        else:
            historical_df = pd.read_csv(dataset.file.path, parse_dates=['timestamp'])

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

            number_of_lagged_features = dataset.input_values

            df = add_lagged_values(df, number_of_lagged_features)

            X_train, X_valid, y_train, y_valid = split(df)

            selected_features = extract_important_features(X_train, y_train)

            X_train_selected = X_train[selected_features]

            lagged_df = pd.DataFrame(lagged_values)

            lagged_df['value'] = lagged_df['value'].fillna(lagged_df['value'].mean())

            lagged_df = lagged_df.loc[:, ['time', 'value']]

            if pd.isna(lagged_df['time'].iloc[0]):
                lagged_df['time'].iloc[0] = (
                        pd.to_datetime(lagged_df['time'].bfill()).iloc[0] - time_difference_train
                )

            if lagged_df['time'].isna().any():
                nan_rows = lagged_df['time'].isna()
                lagged_df.loc[nan_rows, 'time'] = (
                        pd.to_datetime(lagged_df['time'].ffill()) + time_difference_train
                )

            last_timestamp = lagged_df['time'].iloc[-1]

            last_timestamp_datetime = pd.to_datetime(last_timestamp)

            time_difference = pd.to_datetime(lagged_df['time'].iloc[1]) - pd.to_datetime(
                lagged_df['time'].iloc[0])

            next_timestamp = last_timestamp_datetime + time_difference

            new_row = pd.DataFrame({'time': [next_timestamp], 'value': [None]})

            lagged_df = pd.concat([lagged_df, new_row], ignore_index=True)

            lagged_df['value'].iloc[-1] = lagged_df['value'].mean()

            test_df = create_features(lagged_df)
            test_df = add_lagged_values(test_df, number_of_lagged_features)

            y_test = test_df["value"]
            feature_cols = [col for col in test_df.columns if col != "value"]
            feature_cols = [col for col in feature_cols if col != "time"]
            x_test = test_df[feature_cols]
            X_test_selected = x_test[selected_features]

            y_predictions = xgb_model_predictions(X_train_selected, y_train, X_test_selected)

            predicted_next_value = y_predictions[-1].item()

            return JsonResponse({'prediction': predicted_next_value})
