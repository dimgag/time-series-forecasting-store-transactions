# Utilities functions for the project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import itertools




# ARIMA model with best parameters
def tuned_arima(data, diff=0, store=None):
    best_aic = np.inf
    best_pdq = None

    p = q = range(0, 10)
    d = [diff]

    pdq = list(itertools.product(p, d, q))


    for param in pdq:
        try:
            model_arima = ARIMA(data, order=param)
            model_arima_fit = model_arima.fit()
            if model_arima_fit.aic < best_aic:
                best_aic = model_arima_fit.aic
                best_pdq = param

        except Exception as e:
            print(f"Error: {e} with parameters {param}")
            continue
        
    print(f'Best ARIMA parameters: {best_pdq} with AIC: {best_aic}')


    # Create Future dataframe starting from the last date of the dataset
    future_df = pd.DataFrame(index=pd.date_range(start=data.index[-1], periods=50, freq='D'), columns=['n_transactions'])
    
    future_df.sort_index(inplace=True)

    # train arima model with best parameters
    model = ARIMA(data, order=best_pdq).fit()
    predict = model.predict()
    # actual = store['n_transactions']

    # forecast the next 50 days
    forecast = model.forecast(steps=50)

    forecast.index = future_df.index
    future_df['n_transactions'] = forecast

    # save future_df to csv for each store
    future_df.to_csv(f'forecast/arima_forecast_{store}.csv')

    # save logs of evaluation metrics for each store
    mse = mean_squared_error(data, predict)
    mae = mean_absolute_error(data, predict)
    mape = mean_absolute_percentage_error(data, predict)
    rmse = np.sqrt(mse)
    aic = model.aic
    bic = model.bic

    # save logs as .log file
    with open(f'forecast/arima_logs_{store}.log', 'w') as f:
        f.write(f'MSE: {mse}\n')
        f.write(f'MAE: {mae}\n')
        f.write(f'MAPE: {mape}\n')
        f.write(f'RMSE: {rmse}\n')
        f.write(f'AIC: {aic}\n')
        f.write(f'BIC: {bic}\n')
