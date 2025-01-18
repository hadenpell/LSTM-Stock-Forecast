# Predicting TSLA Stock Prices with LSTM 
# Background
LSTM (Long Short Term Memory) is a Recurrent Neural Network (RNN) that is known to do well on time series problems. After taking the Coursera TensorFlow Developer courses, I wanted to attempt my own LSTM project and apply the model to financial time series data as a way to better understand its inner workings. 

I am fully aware that the stock market is volatile and generally unpredictable. A simple LSTM is unlikely to be able to reliably forecast stock prices, especially if it's not also analyzing market sentiment or external events. 

However, in my quest to learn how to build and train LSTMs, I thought I would pick a dataset from a domain that I was interested in to help me master the concepts.

A more thorough breakdown of how I completed this project is available on Medium: https://medium.com/python-in-plain-english/tesla-stock-prediction-with-lstm-325457b80e7b

# Dataset
Using the yahoo finance API, I queried 5 years worth of daily stock data (Stock closing price) for Tesla (TSLA) stock, from December 2018 to December 2023.
I saved this data as a CSV file and loaded it each time instead of calling the API each time so I'd work with a constant dataset.

# Objective
To see if I could reliably predict the next 30 days of TSLA stock prices using an LSTM model.

# Approach

## Baseline
Before fitting the final LSTM model, I first ran a moving average (MA) baseline model on the time series data and recorded a baseline RMSE.

## Data preprocessing
Data was split into train, validation and test sets and scaled them using StandardScaler. Then, I had to convert the data into windowed sequences to be compatible with the LSTM model.

LSTMs process data in sequences, which essentially means a window of historical data. There are some data structures in Tensorflow that can ease the process of converting a standard dataset, with just a dataframe or array of time stamps and values, into sequences.
The tf.data.Dataset.from_tensor_slices function creates a Tensor for each item in the original numpy array. Once the numpy array is in a Dataset format, there are a variety of helpful methods that can be called on it.

My windowed_dataset method (most of which was adapted from the TensorFlow Developer course) does the following:
* Generates a TensorFlow dataset from the inputted series values (using from_tensor_slices method)
* Windows the dataset using dataset.window method to create windows of a certain window_size - and added +1 to the window size because not only did I want to window the X, I also wanted to grab the y values so that I could later separate the X and y
* .window produces values like this: [1, 2, 3, 4] where 1,2, and 3 are the features and 4 is the y; but tensorflow doesn’t want data in this list format. It wants an object that looks more like this: (tensor=[1 2 3], tensor=4), where it stores the X as an array in one tensor and the y in another tensor in the form of a tuple. To get this format, I use .flat_map to turn [1, 2, 3, 4] into a [1 2 3 4] followed by .map to separate into X and y.
* Shuffles windows using the .shuffle method, to avoid overfitting
* Creates batches of windows; this way, the model will learn from multiple sequences at a time instead of just one.

I called the windowed_dataset method on both the train dataset and validation dataset.

## Model architecture
Two LSTM layers tend to do generally well for modelling complex time series data sets. Return_sequences=True ensures that the output from the first LSTM layer gets passed on to the second one. The last LSTM layer should not return sequences. I did try adding in a third LSTM layer, but two layers had better performance.

When it comes to the number of units, I started with 50. A good range is 32–128, but it depends on the problem and data. I also tried 128, 64, and 60 units in various combinations, but found that 50 in both layers worked generally well.
In Dropout layers, a proportion of connections to LSTM units are excluded from training updates to prevent overfitting. I set my dropout layers to 0.2, which means that 20% of units are “dropped” (excluded). A general rule of thumb is to include a Dropout layer after each LSTM layer.

## Tuning the learning rate
I created a model specifically for tuning the learning rate and plotted the results. Based on the chart it appeared that the optimal learning rate was 0.001.
![alt text](https://github.com/hadenpell/LSTM-Stock-Forecast/blob/main/learning_rate.png?raw=True)

## Train model
I repeated the same steps to build and compile my final model with the chosen learning rate, except for a few changes:
* I added 2 new callbacks and excluded the learning rate. These callbacks were ModelCheckpoint and EarlyStopping.
* EarlyStopping controls the number of epochs the model will train for and stops the training process if the model stops improving after {patience} amount of consecutive epochs. I set patience to 10.
* ModelCheckpoint saves the best model as an object in my directory, based on the validation loss, across all epochs that were run. I can load this model from memory at any time to make future predictions.

## Predictions
I prepared the test set the same way I windowed and batched my train and validation sets. The main difference is that I did not shuffle the data or separate the features and labels. Because of this, I created a window with my window_size only and not window_size + 1 to only include the time series actual values for each window and not their corresponding next value.
Then I loaded the saved model (/model1) and predicted on the test set. To get the actual prediction values, I called reverse_transform on the scaled output. The test set RMSE was 8.46.
![alt text](https://github.com/hadenpell/LSTM-Stock-Forecast/blob/main/test_preds.png?raw=True)

I repeated the same process for the train set to check for overfitting, and got a train RMSE of about 4. It’s better than the test set, as expected, but not enough of a discrepancy to be concerned about overfitting.

To forecast into the future on completely unseen data (30 days), I had to use a shifting window approach since after the first prediction, there would be no more actual historical data to use as features.

I approached the forecast as follows:
* Retrieve the most recent 20 values and predict 1 day ahead.
* With that new forecasted value, shift the window by 1. The most recent 20 values are now updated with 1 prediction counting as a “past value”.
* Continue to shift the window using predictions as features until I have 30 days of predictions.

Here was the final forecast:
![alt text](https://github.com/hadenpell/LSTM-Stock-Forecast/blob/main/final_fcst.png?raw=True)

# Conclusions + Drawbacks
The final forecast didn't look great. Though the values were within a reasonable range for what you'd expect, the forecast remained fairly stable with a slight upward trend of ... not even one dollar.

As the model predicts out farther and farther, uncertainty compounds. It’s important to consider this when choosing how many days ahead you want to forecast and how seriously you want to take each prediction.

I built this model to only predict ahead 1 day at a time, for 30 days. An alternative would be to train the model to actually forecast 30 days at a time (multi-step time series forecasting), which may have mitigated some of the compounding error issues and instead of using predictions to forecast would have predicted a 30-day sequence all at once based on the prior 20 days. This would require a final Dense layer of size 30 rather than 1 in my model architecture. 

Additionally, as I stated in the Background section, a simple LSTM with only historical data as a feature is not ideal for forecasting the stock market, which is a highly volatile, often random time series dataset.

However, this project allowed me to learn about how to format time series data for LSTM input, how to compile, train, and tune an LSTM, and work with the TensorFlow library, so it was a success on those fronts.
