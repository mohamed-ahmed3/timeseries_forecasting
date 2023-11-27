from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


def rf_model_predictions(x_train, y_train, x_valid):
    model = RandomForestRegressor()

    model.fit(x_train, y_train)
    y_predicted = model.predict(x_valid)

    return y_predicted


def linear_model_predictions(x_train, y_train, x_valid):
    model = LinearRegression()

    model.fit(x_train, y_train)
    y_predicted = model.predict(x_valid)

    return y_predicted


def xgb_model_predictions(x_train, y_train):
    model = xgb.XGBRegressor()

    model.fit(x_train, y_train)

    return model
