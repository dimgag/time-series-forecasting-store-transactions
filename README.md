# Albert Heijn Data Science N-Forecasting Case Study

## Context
The unavailability of products in our stores (due to stockouts) is a main driver for customer dissatisfaction and lost sales. Simultaneously, overstocking of our products contributes to food waste (due to spoilage), incurring high costs and causing environmental harm. Therefore, it is crucial for Albert Heijn to obtain reliable estimates on the daily number of customers in each store in the time to come. This helps us to optimize:

- replenishment decisions (i.e., deciding how much of each product to send to each store over time), and
- staffing decisions (i.e., how many employees to schedule per store to make sure the store operations run smoothly).

This assignment is about forecasting the number of transactions that take place each day in each store. We call this the “N-Forecast”. N is close to the number of customers that come to a store but not the same. While most customers visit the store at most once a day, there are exceptions where some might come multiple times, leading to multiple transactions being counted. Additionally, there could be instances where customers leave the store without making any transactions.

## Assignment details
Your task is to develop a Python solution that forecasts the number of transactions per day per store up to 50 days ahead. For example, if you run your solution on 1 January 2023, it is expected to forecast for each store the number of transactions on 2 January 2023, 3 January 2023, …, 19 February 2023.

We expect you to design your own model training and validation setup (this includes choosing an appropriate error metric for your forecasts). It is important for you to be able to convince us (your stakeholders) that your model is carefully designed and can generate good forecasts going forward.

## Deliverables
- A python Jupyter Notebook where we can see the results of your work, and follow how you approached the problem. Not all your code has to be in the notebook, you can also use scripts or a package to build your solution, but we want to be able to see the results easily. Please submit all of your code.
- A short exploratory data analysis highlighting important characteristics of the data or problems you find in the data
- Model evaluation with an analysis of the performance
- Some comments on possible future improvements you would make if you worked further on this project

## Data
The dataset has been derived from Albert Heijn data but manipulated and obfuscated in a way as to preserve commercially sensitive information. It is provided as a single parquet file: n_forecast.parquet

You are provided with an historical dataset containing the following fields:

- `sales_date`: The date on which the transactions occurred
- `store_hashed`: a hashed version of the store number
- `n_transactions`: The number of transactions that took place in that store on that day - the target of the prediction in the assignment
- `store_format`: An categorical feature indicating that the store has a particular type (e.g. To Go or XL)
- `zipcode_region`: The first two digits of the zipcode where the store is located
- `region`: The holiday region in which the store falls
- A number of binary features indicating whether a particular date falls on a national holiday or a regional school vacation period.
- `datetime_store_open`: The time at which the store opened on the sales_date
- `datetime_store_closed`: The time at which the store closed on the sales_date

You may add other public datasets if you wish to improve the forecast (e.g. weather related data) – this is however not required.

| File Name | Description |
| --- | --- |
| `dg_preprocessing_EDA.ipynb` | Data Preprocessing & Exploratory Data Analysis|
|`dg_TSA_TSF.ipynb` | Time Series Analysis & Forecasting |
