from django.db import models


class TimeSeriesDatasets(models.Model):
    dataset_id = models.IntegerField()
    file = models.FileField()
    input_values = models.IntegerField(editable=False, null=True)
