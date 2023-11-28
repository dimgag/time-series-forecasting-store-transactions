# Utilities functions for the project
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.model_selection import ParameterGrid
import itertools




# Stationarity test ADF
def test_stationarity_adf(timeseries):
        # Determing rolling statistics
        rolmean = timeseries.rolling(12).mean()
        rolstd = timeseries.rolling(12).std()
        
        # Plot rolling statistics:
        fig = plt.figure(figsize=(12, 8))
        orig = plt.plot(timeseries, color='blue',label='Original')
        mean = plt.plot(rolmean, color='red', label='Rolling mean')
        std = plt.plot(rolstd, color='black', label = 'Rolling std')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
        plt.show()
        
        # Perform Dickey-Fuller test:
        print('Results of Dickey-Fuller Test:')
        dftest = adfuller(timeseries, autolag='AIC', maxlag=12)
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
        
        if (dftest[1] < 0.05) and (dftest[0] < dftest[4]['5%']): # add also the condition of the test statistic being lower than the critical value
            print('The p-value is lower than 0.05, the null hypothesis is rejected and the data is stationary')
        else:
            print('The p-value is higher than 0.05, the null hypothesis is accepted and the data is non-stationary')

        print(dfoutput)

# Stationarity test KPSS
def test_stationarity_kpss(timeseries):
        # Determing rolling statistics
        rolmean = timeseries.rolling(12).mean()
        rolstd = timeseries.rolling(12).std()
        
        # Plot rolling statistics:
        fig = plt.figure(figsize=(12, 8))
        orig = plt.plot(timeseries, color='blue',label='Original', marker='.', linestyle='None')
        mean = plt.plot(rolmean, color='red', label='Rolling mean', marker='.', linestyle='None')
        std = plt.plot(rolstd, color='black', label = 'Rolling std', marker='.', linestyle='None')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
        plt.show()
        
        # Perform KPSS test:
        print('Results of KPSS Test:')
        kpsstest = kpss(timeseries, regression='c', nlags="auto")
        kpssoutput = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
        
        for key,value in kpsstest[3].items():
            kpssoutput['Critical Value (%s)'%key] = value
        
        print(kpssoutput)


# Test stationarity for stores
def test_stores_stationarity(stationary_data, plot=False, results=False):
    """
    Test the stationarity of time series data for each store in a given dataset.

    Parameters:
    -----------
    stationary_data : pandas.DataFrame
        A DataFrame containing time series data for multiple stores, with columns
        'store_hashed' (store identifier) and 'n_transactions' (number of transactions).
    plot : bool, optional
        Whether to plot the rolling mean and standard deviation for each store (default False).

    Returns:
    --------
    None
        The function prints the results of the Dickey-Fuller test for each store,
        and the number of stationary and non-stationary stores found.
    """
    rolmean = {}
    rolstd = {}
    stationary = []
    non_stationary = []

    for store in stationary_data['store_hashed'].unique():
        store_data = stationary_data[stationary_data['store_hashed'] == store]
        
        rolmean[store] = store_data['n_transactions'].rolling(12).mean()
        rolstd[store] = store_data['n_transactions'].rolling(12).std()


        if plot == True:
            fig = plt.figure(figsize=(12, 8))
            orig = plt.plot(store_data['n_transactions'], color='blue',label='Original')
            mean = plt.plot(rolmean[store], color='red', label='Rolling mean')
            std = plt.plot(rolstd[store], color='black', label = 'Rolling std')
            plt.legend(loc='best')
            plt.title('Rolling Mean & Standard Deviation')
            plt.show()


        dftest = adfuller(store_data['n_transactions'], autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value

        if dftest[1] < 0.05 and dftest[0] < dftest[4]['5%']:
            stationary.append(store)
        else:
            non_stationary.append(store)
        
        if results == True:
            print('Results of Dickey-Fuller Test for store {}:'.format(store))
            print(dfoutput)
            print('\n')

    print('Stationary stores:', len(stationary))
    print('Non stationary stores:', len(non_stationary))
    return stationary, non_stationary





def differencing(data, non_stationary_idx):
    # Function to apply differencing in time series data to make them stationary
    stationary = []
    non_stationary = []
    for idx in non_stationary_idx:
        store_data = data[data['store_hashed'] == idx]
        
        # Apply differencing 
        diff = store_data['n_transactions'].diff().dropna()

        # Run ADF test
        dftest = adfuller(diff, autolag='AIC')
        if dftest[1] < 0.05 and dftest[0] < dftest[4]['5%']:
            # print(f'Store {idx} is now stationary')
            data.loc[data['store_hashed'] == idx, 'n_transactions'] = diff
            stationary.append(idx)

        else:
            # print(f'Store {idx} is still non-stationary')
            non_stationary.append(idx)
    
    print('Stationary stores:', len(stationary))
    print('Non stationary stores:', len(non_stationary))
    return stationary, non_stationary



# Find best parameters for ARIMA model

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
        
    print(f'Best ARIMA parameters: {best_pdq} with AIC: {best_aic}')
    return best_pdq


def arima_hyperparameters_grid_search(data, diff=0):
    best_aic = np.inf
    best_pdq = None

    # Define the hyperparameters
    hyperparameters = {'p': range(0, 10), 'd': [diff], 'q': range(0, 10)}

    # Create the grid
    grid = ParameterGrid(hyperparameters)

    for params in grid:
        try:
            model_arima = ARIMA(data, order=(params['p'], params['d'], params['q']))
            model_arima_fit = model_arima.fit()
            if model_arima_fit.aic < best_aic:
                best_aic = model_arima_fit.aic
                best_pdq = (params['p'], params['d'], params['q'])
        except Exception as e:
            print(f"Error: {e} with parameters {params}")
            continue
        
    print(f'Best ARIMA parameters: {best_pdq} with AIC: {best_aic}')
    return best_pdq


def train_arima_model(data, pdq):
    arima_model = ARIMA(data, order=pdq)
    arima_model_fit = arima_model.fit()
    return arima_model_fit

def forecast_arima(model_fit, steps=50):
    future = pd.DataFrame(index=pd.date_range(start='2021-03-28', periods=50, freq='D'), columns=['n_transactions'])
    future.sort_index(inplace=True)

    forecast = model_fit.forecast(steps=steps)
    forecast.index = future.index
    future['n_transactions'] = forecast
    
    # actual = store['n_transactions']

    return forecast[0]