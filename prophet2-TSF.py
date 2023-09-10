# Prophet model with additional regressor the working_hours of each store
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
sns.set_style('whitegrid')

# Import the data
data = pd.read_parquet('n_forecast_preprocessed.parquet', engine='pyarrow', dtype_backend='numpy_nullable')

# Convert the 'sales_date' column to a datetime type
data['sales_date'] = pd.to_datetime(data['sales_date'])

# Group data by 'store_hashed' and 'sales_date' to get daily transaction counts per store
daily_sales = data.groupby(['store_hashed', 'sales_date'])[['n_transactions', 'working_hours']].sum().reset_index()

# Rename columns for Prophet's input format
daily_sales.rename(columns={'sales_date': 'ds', 'n_transactions': 'y'}, inplace=True)

# Initialize Prophet models for each store with custom regressors
models = {}
for store in daily_sales['store_hashed'].unique():
    store_data = daily_sales[daily_sales['store_hashed'] == store]
    
    # Create a Prophet model with additional regressor 'working_hours'
    model = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=False)
    model.add_regressor('working_hours')
    
    # Add the regressor data to the model
    store_data.rename(columns={'working_hours': 'working_hours'}, inplace=True)
    model.fit(store_data)
    models[store] = model

# Create a dataframe for future dates for forecasting (50 days ahead)
future_dates = pd.DataFrame({'ds': pd.date_range(start=daily_sales['ds'].max(), periods=51, freq='D')[1:]})

# Initialize metrics storage
mape_scores = {}
rmse_scores = {}

# Forecast for each store and calculate metrics
forecast_data = pd.DataFrame(columns=['store_hashed', 'ds', 'yhat'])
for store, model in models.items():
    future = model.predict(future_dates)
    future['store_hashed'] = store
    forecast_data = pd.concat([forecast_data, future[['store_hashed', 'ds', 'yhat']]])
    
    # Calculate MAPE
    actual = store_data.set_index('ds')['y']
    predicted = future.set_index('ds')['yhat']
    mape = np.mean(np.abs((actual - predicted) / actual)) * 100
    mape_scores[store] = mape

    # Calculate RMSE with common dates
    common_dates = actual.index.intersection(predicted.index)

    if not common_dates.empty:
        rmse = np.sqrt(mean_squared_error(actual.loc[common_dates], predicted.loc[common_dates]))
        rmse_scores[store] = rmse
    else:
        print(f"No common dates found for Store {store}. Skipping RMSE calculation.")





# Plot the time series for each store
plt.figure(figsize=(12, 6))
for store, store_data in forecast_data.groupby('store_hashed'):
    plt.plot(store_data['ds'], store_data['yhat'], label=f'Store {store}')
plt.xlabel('Date')
plt.ylabel('Forecasted Transactions')
plt.title('Forecasted Transactions per Store')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()

# Print MAPE and RMSE scores
for store, mape in mape_scores.items():
    print(f'Store {store} MAPE: {mape:.2f}%')
    
for store, rmse in rmse_scores.items():
    print(f'Store {store} RMSE: {rmse:.2f}')
