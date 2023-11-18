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
            historical_df = historical_df.ffill()

            df = create_features(historical_df)

            number_of_lagged_features = dataset.input_values

            df = add_lagged_values(df, number_of_lagged_features)

            X_train, X_valid, y_train, y_valid = split(df)

            selected_features = extract_important_features(X_train, y_train)

            X_train_selected = X_train[selected_features]

            lagged_df = pd.DataFrame(lagged_values)

            lagged_df = lagged_df.ffill()

            last_timestamp = lagged_df['timestamp'].iloc[-1]

            last_timestamp_datetime = pd.to_datetime(last_timestamp)

            time_difference = pd.to_datetime(lagged_df['timestamp'].iloc[1]) - pd.to_datetime(
                lagged_df['timestamp'].iloc[0])

            next_timestamp = last_timestamp_datetime + time_difference

            new_row = pd.DataFrame({'timestamp': [next_timestamp], 'value': [None]})

            lagged_df = pd.concat([lagged_df, new_row], ignore_index=True)

            lagged_df['value'].iloc[-1] = lagged_df['value'].mean()

            test_df = create_features(lagged_df)
            test_df = add_lagged_values(test_df, number_of_lagged_features)

            y_test = test_df["value"]
            feature_cols = [col for col in test_df.columns if col != "value"]
            feature_cols = [col for col in feature_cols if col != "timestamp"]
            x_test = test_df[feature_cols]
            X_test_selected = x_test[selected_features]

            y_predictions = xgb_model_predictions(X_train_selected, y_train, X_test_selected)

            predicted_next_value = y_predictions[-1].item()

            return JsonResponse({'prediction ': predicted_next_value})
