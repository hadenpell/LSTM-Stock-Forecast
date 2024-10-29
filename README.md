# Predicting TSLA Stock Prices with LSTM 
LSTM (Long Short Term Memory) is a Recurrent Neural Network (RNN) that is known to do well on time series problems. I wanted to apply this to financial time series data as a way to better understand the inner workings of this model. 

I am fully aware that the stock market is volatile and generally unpredictable. A simple LSTM is unlikely to be able to reliably forecast stock prices, especially if it's not analyzing market sentiment or external events. I simply wanted to create a model using a dataset and work in a domain that I was interested in to help me master the concepts.

# Dataset
Using the yahoo finance API, I queried 5 years worth of daily stock data (Stock closing price) for Tesla (TSLA) stock, from December 2018 to December 2023.

# Objective
To see if I could reliably predict the next 30 days of TSLA stock prices using an LSTM model.

# Approach
