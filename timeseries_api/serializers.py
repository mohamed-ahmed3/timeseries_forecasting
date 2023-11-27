from rest_framework import serializers
from .models import *


class TimeSeriesSerializer(serializers.ModelSerializer):
    class Meta:
        model = TimeSeriesDatasets
        fields = [
            'dataset_id',
            'file',
            'input_values'
        ]


class SelectedFeaturesSerializer(serializers.ModelSerializer):
    class Meta:
        model = SelectedFeature
        fields = [
            'feature_name'
        ]
