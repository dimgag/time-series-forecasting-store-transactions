# Utilities functions for the project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss


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
        
        if dftest[1] < 0.05:
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
def test_stores_stationarity(stationary_data, plot=False):
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
    stationary_stores = []
    non_stationary_stores = []

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

        # print('Results of Dickey-Fuller Test:')
        dftest = adfuller(store_data['n_transactions'], autolag='AIC', maxlag=12)
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])

        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value

        if dftest[1] < 0.05:
            # print('The p-value is lower than 0.05, the null hypothesis is rejected and the data is stationary')
            stationary_stores.append(store)
        else:
            # print('The p-value is higher than 0.05, the null hypothesis is accepted and the data is non-stationary')
            non_stationary_stores.append(store)
        
    print('Stationary stores:', len(stationary_stores))
    print('Non stationary stores:', len(non_stationary_stores))
    return stationary_stores, non_stationary_stores