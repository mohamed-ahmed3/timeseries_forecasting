from statsmodels.tsa.arima.model import ARIMA


def ARIMA_model_prediction(dataframe):
    model = ARIMA(dataframe['value'], order=(1, 0, 5))
    model = model.fit()

    forecast_steps = 1
    forecast = model.get_forecast(steps=forecast_steps)

    predicted_value = forecast.predicted_mean.iloc[-1]

    return predicted_value
