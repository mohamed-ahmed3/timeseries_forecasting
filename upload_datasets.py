import os
from timeseries_api.models import *
from lagged_features_extraction import determine_lagged_features


def upload_all_datasets(directory_path):
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(directory_path, filename)
            id = ''.join(filter(str.isdigit, filename))

            dataset, created = TimeSeriesDatasets.objects.get_or_create(
                dataset_id=id,
                file=file_path
            )
            if created:
                determine_lagged_features(file_path)


