import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller

import warnings
import itertools
warnings.filterwarnings('ignore')



# Stationarity test for all stores...
# let's make a class for this.
class StationarityTest:
    # initialize the class
    def __init__(self, series):
        self.series = series
        self.p_value = None
        self.is_stationary = None
        self.test_statistic = None
        self.critical_values = None
        self.significance_level = 0.05
        self.n_lags = None
        self.p_value = None
        self.test_statistic = None
        self.critical_values = None
    
    def adfuller_test(self):
        '''Augmented Dickey-Fuller test for stationarity'''
        print('Augmented Dickey-Fuller Test: ')
        adf_test = adfuller(self.series, autolag='AIC')
        self.n_lags = adf_test[2]
        self.p_value = adf_test[1]
        self.test_statistic = adf_test[0]
        self.critical_values = adf_test[4]
        print(f'ADF Statistic: {self.test_statistic}')
        print(f'p-value: {self.p_value}')
        print(f'No. of lags: {self.n_lags}')
        print('Critical Values: ')
        for key, value in self.critical_values.items():
            print(f'\t{key}: {value}')
        if self.p_value < self.significance_level:
            self.is_stationary = True
            print('The series is stationary')
        else:
            self.is_stationary = False
            print('The series is not stationary')
        return self.is_stationary


    def make_stationary(self):
        '''
        Apply differencing to make the series stationary
        and return the differenced series and the differencing order d
        if the series is already stationary, return the original series and d = 0
        if the series are not stationary apply differencing until the series is stationary
        and return the differenced series and the differencing order d
        '''
        if self.is_stationary:
            return self.series, 0
        else:
            d = 0
            while not self.is_stationary:
                d += 1
                self.series = self.series.diff().dropna()
                self.adfuller_test()
            return self.series, d

    def invert_differencing(self, differenced_series, d):
        '''Invert differencing
        return the original series from the differenced series
        and the differencing order d
        if d = 0, return the differenced series
        if d > 0, return the original series from the differenced series'''
        
        if d == 0:
            return differenced_series
        else:
            inverted_series = [differenced_series[0]]
            for i in range(1, len(differenced_series)):
                inverted_value = differenced_series[i] + inverted_series[i-1]
                inverted_series.append(inverted_value)
            return inverted_series
        
    
    def find_best_arima_params(self):
        '''Apply find the best parameters for ARIMA model
        and return the best parameters for each store'''
        
        best_aic = np.inf
        best_pdq = None

        p = q = range(0, 10)
        d = [1]
        pdq = list(itertools.product(p, d, q))

        for param in pdq:
            try:
                model_arima = ARIMA(self.series, order=param)
                model_arima_fit = model_arima.fit()
                if model_arima_fit.aic < best_aic:
                    best_aic = model_arima_fit.aic
                    best_pdq = param
            except:
                continue

        return best_pdq
         
    
    def apply_arima(self, order):
        '''Apply ARIMA model to the series
        and return the predictions and the residuals'''

        model_arima = ARIMA(self.series, order=order)
        model_arima_fit = model_arima.fit()
        predictions = model_arima_fit.predict()
        residuals = model_arima_fit.resid
        return predictions, residuals
    
    def forecast(self, n_days):
        '''Forecast for next n_days using the ARIMA model above'''
        best_pdq = self.find_best_arima_params()
        predictions, residuals = self.apply_arima(best_pdq)
        forecast = predictions[-n_days:]
        return forecast, residuals
    
    def plot_predictions(self, n_days):
        '''Plot the predictions and actual values'''
        predictions, residuals = self.forecast(n_days)
        plt.figure(figsize=(12, 6))
        plt.title('Predictions')
        plt.plot(predictions)
        plt.plot(self.series[-n_days:])
        plt.show()


    def plot_forecast(self, n_days):
        '''plot the forecast and the actual values'''
        forecast, residuals = self.forecast(n_days)
        plt.figure(figsize=(12, 6))
        plt.title('Forecast')
        plt.plot(forecast)
        plt.plot(self.series[-n_days:])
        plt.show()



# # Apply the class to each store in the data. Note you have to create the future dataframe for the next n_steps.
# # Example:
# for store_id in store_ids:
#     print(f'Store: {store_id}')
#     store = df[df['store_id'] == store_id]
#     stationarity_test = StationarityTest(store['n_transactions'])
#     stationarity_test.adfuller_test()
#     stationary_series, d = stationarity_test.make_stationary()
#     print(f'Differencing order: {d}')
#     inverted_series = stationarity_test.invert_differencing(stationary_series, d)
#     stationarity_test.plot_predictions(30)
#     stationarity_test.plot_forecast(30)
#     print('\n\n')

# TBC...