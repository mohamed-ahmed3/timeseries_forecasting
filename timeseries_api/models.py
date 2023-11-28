from django.db import models


class TimeSeriesDatasets(models.Model):
    dataset_id = models.IntegerField()
    file = models.FileField()
    input_values = models.IntegerField(editable=False, null=True)
    model = models.BinaryField(null=True)
    selected_features = models.JSONField(null=True, blank=True)

