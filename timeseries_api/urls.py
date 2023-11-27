from django.urls import path
from .views import *

urlpatterns = [
    path('datasets/', ListDatasets.as_view(), name="ListDatasets"),
    path('datasets/creation', UploadDatasets.as_view(), name="UploadDatasets"),
    path('models/creation', SaveModel.as_view(), name="UploadModels"),
    path('datasets/prediction', ForecastPrediction.as_view(), name="Predict")
]