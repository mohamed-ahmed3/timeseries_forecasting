# timeseries_forecasting
Forecasting time series data using different models, including linear regression, random forests, and XGBoost.

## Seting up the container
1- Clone the github repository.
2- Open a terminal in the directory containing the repository.
3- Write the command: docker compose up

## Getting the datasets with their corresponding input values
To list all the datasets with their corresponding lagged values, send a get request with the following url: http://localhost:8086/timeseries_api/datasets/

## Predicting the next value
To predict the next value of the given input values, send a post request to the following url: http://localhost:8086/timeseries_api/datasets/prediction
