from django.db import models


class TimeSeriesDatasets(models.Model):
    dataset_id = models.IntegerField()
    file = models.FileField()
    input_values = models.IntegerField(editable=False, null=True)
    model = models.BinaryField(null=True)


class SelectedFeature(models.Model):
    feature_name = models.CharField(max_length=255)
