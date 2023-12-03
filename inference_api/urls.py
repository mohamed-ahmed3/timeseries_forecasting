from django.urls import path
from .views import *

urlpatterns = [
    path('inference/', Mlflow.as_view(), name="Inference"),
]