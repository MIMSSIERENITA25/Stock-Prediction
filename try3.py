#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_datareader import data as pdr
import yfinance as yf
from datetime import datetime

# Fetch data for each company and store it in company_list
tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']
company_list = []
for symbol in tech_list:
    company_data = yf.download(symbol, start='2023-01-01', end='2024-01-01')
    company_list.append(company_data)

# Load stock data
def load_data():
    tech_list = ['AAPL', 'GOOG', 'MSFT', 'AMZN']
    end = datetime.now()
    start = datetime(end.year - 1, end.month, end.day)
    for stock in tech_list:
        globals()[stock] = yf.download(stock, start, end)
    company_list = [AAPL, GOOG, MSFT, AMZN]
    company_name = ["APPLE", "GOOGLE", "MICROSOFT", "AMAZON"]
    for company, com_name in zip(company_list, company_name):
        company["company_name"] = com_name
    df = pd.concat(company_list, axis=0)
    return df

df = load_data()

# Set page title
st.title('Stock Market Analysis Web App')

# Add sidebar navigation
page = st.sidebar.radio("Navigation", ["Recent Stock Data", "Summary Statistics", "Moving Averages", "Daily Returns", "Correlation", "Stock Price Prediction"])

# Based on the selected page, display the corresponding content
if page == "Recent Stock Data":
    st.write("### Recent Stock Data")
    st.write(df.tail(10))

elif page == "Summary Statistics":
    st.write("### Summary Statistics")
    st.write(df.describe())

elif page == "Moving Averages":
    st.write("### Moving Averages")
    # Define the moving averages you want to calculate
    ma_day = [10, 20, 50]

    # Calculate moving averages for each company
    for ma in ma_day:
        for company in company_list:
            column_name = f"MA for {ma} days"
            company[column_name] = company['Adj Close'].rolling(window=ma).mean()

    # Display the moving average plots
    for ma in ma_day:
        plt.figure(figsize=(10, 6))
        for i, (company, company_name) in enumerate(zip(company_list, ["APPLE", "GOOGLE", "MICROSOFT", "AMAZON"]), 1):
            plt.subplot(2, 2, i)
            company[f'MA for {ma} days'].plot()
            plt.title(f"{company_name} - Moving Average for {ma} Days")
            plt.xlabel('Date')
            plt.ylabel('Adj Close')
            plt.grid(True)
        plt.tight_layout()
        st.pyplot()

elif page == "Daily Returns":
    st.write("### Daily Returns")
    # Calculate daily returns
    for company in company_list:
        company['Daily Return'] = company['Adj Close'].pct_change()

    # Plot daily return percentage
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 10))

    for i, company in enumerate(company_list, 1):
        plt.subplot(2, 2, i)
        company_name = company.index[0]  # Get the company name from the index of the DataFrame
        company['Daily Return'].hist(bins=50)
        plt.xlabel('Daily Return')
        plt.ylabel('Counts')
        plt.title(f'{company_name}')

    fig.tight_layout()

    # Display the plot in the Streamlit app
    st.pyplot(fig)

elif page == "Correlation":
    st.write("### Correlation Between Stocks")
    # Select only numeric columns for correlation calculation
    numeric_df = df.select_dtypes(include=np.number)
    # Calculate correlation matrix
    corr = numeric_df.corr()

    # Plot heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=.5)
    plt.title("Correlation Heatmap")
    st.pyplot()

elif page == "Stock Price Prediction":
    st.write("### Stock Price Prediction")

    # Placeholder for prediction code
    # Assuming you have already trained a model and made predictions
    # Replace this with your actual prediction data
    predicted_prices = [100, 110, 115, 120, 125, 130, 135]

    # Plotting the predicted prices
    plt.figure(figsize=(10, 6))
    plt.plot(predicted_prices, marker='o', color='blue', linestyle='-')
    plt.title('Predicted Stock Prices')
    plt.xlabel('Days')
    plt.ylabel('Price')
    st.pyplot()


# In[ ]:




