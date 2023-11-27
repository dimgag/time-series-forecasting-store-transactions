# Utilities functions for the project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import itertools

stationary_stores = []
stationary_stores_2 = []
# Create empty dataframe with pdq values
store_params = pd.DataFrame(columns=['store', 'pdq'])


def arima_hyperparameters(data, diff=0):
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
        
    # print(f'Best ARIMA parameters: {best_pdq} with AIC: {best_aic}')
    return best_pdq


# fill the store_params dataframe with the best pdq values for each store
for store in stationary_stores:
    store_params = store_params.append({'store': store, 'pdq': arima_hyperparameters(store['n_transactions'])}, ignore_index=True)

# fill the store_params dataframe with the best pdq values for each store
for store in stationary_stores_2:
    store_params = store_params.append({'store': store, 'pdq': arima_hyperparameters(store['n_transactions'], diff=1)}, ignore_index=True)


# Use the store_params dataframe to fit the ARIMA model for each store and forecast the next 50 days
for index, row in store_params.iterrows():
    # create empty future dataframe for each store
    # create empty future dataframe
    future = pd.DataFrame(index=pd.date_range(start='2021-03-28', periods=50, freq='D'), columns=['n_transactions'])
    future.sort_index(inplace=True)

    store = row['store']
    pdq = row['pdq']
    arima_model = ARIMA(store['n_transactions'], order=pdq)
    arima_model_fit = arima_model.fit()
    
    # Evaluate the model
    mse = mean_squared_error(store['n_transcations'], forecast_arima)
    mae = mean_absolute_error(store['n_transcations'], forecast_arima)
    mape = mean_absolute_percentage_error(store['n_transcations'], forecast_arima)
    rmse = np.sqrt(mse)
    print(f'Store {store["store_hashed"]}')
    print(f'MSE: {mse:.3f}')
    print(f'MAE: {mae:.3f}')
    print(f'MAPE: {mape:.3f}')
    print(f'RMSE: {rmse:.3f}')
    print(f'AIC: {arima_model_fit.aic:.3f}')
    print(f'BIC: {arima_model_fit.bic:.3f}')

    forecast_arima = arima_model_fit.forecast(steps=50)
    store['forecast'] = forecast_arima

    # Plot the forecast
    plt.figure(figsize=(12, 6))
    plt.plot(store['n_transactions'], label='Train')
    plt.plot(store['forecast'], label='Forecast')
    plt.legend()
    plt.show()





