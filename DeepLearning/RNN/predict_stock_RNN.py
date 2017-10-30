#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 09:32:01 2017

@author: Scott
@problem: Predict stock price :D 
"""
# Import librarys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Hide tensorflow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Fix random seed for reproducibility
np.random.seed(7)

# Read dataset
stock_dataframe = pd.read_csv('google_stock_data.csv')
stock_data = stock_dataframe.iloc[:, 4:5].values # just use close price
stock_data = stock_data.astype('float32')

# Normalize stock_data
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
stock_data_scaled = sc.fit_transform(stock_data)

# Split data(Training and Test)
train_size = int(len(stock_data_scaled) * 0.7)
test_size = len(stock_data_scaled) - train_size
train_set, test_set = stock_data_scaled[0:train_size,:], stock_data_scaled[train_size:len(stock_data),:]
print("train_data: %d\ntest_data: %d\n" % (len(train_set), len(test_set)))

# Convert array data to times step data array
# Like this
# t-timesstep+1, t-timesstep+2, ... , t(X),    t+1(Y)
def create_timesteps_dataset(dataset, time_steps=60):
    """
    Arguments:
    dataset -- numpy array of any shape
    time_steps -- the number of previous time steps to use as input data to predict the next step

    Returns:
    A -- numpy array of structed by time teps
    """
    X_data, Y_data = [], []
    for i in range(len(dataset)-time_steps-1):
        timesteps_data = dataset[i:(i+time_steps), 0]
        X_data.append(timesteps_data)
        Y_data.append(dataset[i+time_steps, 0])

    return np.array(X_data), np.array(Y_data)

time_steps=60
X_train, Y_train = create_timesteps_dataset(train_set, time_steps)
X_test, Y_test = create_timesteps_dataset(test_set, time_steps)

# Reshape dataset to LSTM input format [samples, timesteps, feature]
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Create 4 LSTM layer with Dropout
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

model = Sequential()
model.add(LSTM(units=128, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, Y_train, epochs=100, batch_size=16, verbose=2)

# Predict stock
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Invert predict data
train_predict = sc.inverse_transform(train_predict)
Y_train = sc.inverse_transform([Y_train])
test_predict = sc.inverse_transform(test_predict)
Y_test = sc.inverse_transform([Y_test])

# Check RMSE(Root Mean Squared Error)
from sklearn.metrics import mean_squared_error
import math
train_score = math.sqrt(mean_squared_error(Y_train[0], train_predict[:,0]))
test_score = math.sqrt(mean_squared_error(Y_test[0], test_predict[:,0]))
print("Train score: %.2f RMSE\nTest score: %.2f RMSE\n" % (train_score, test_score))

# Shift prediction data for Visualize
train_plot = np.empty_like(stock_data_scaled)
train_plot[:, :] = np.nan
train_plot[time_steps:len(train_predict)+time_steps, :] = train_predict

test_plot = np.empty_like(stock_data_scaled)
test_plot[:, :] = np.nan
test_plot[len(train_predict)+(time_steps*2)+1:len(stock_data_scaled)-1, :] = test_predict

# Visualize data
plt.plot(sc.inverse_transform(stock_data_scaled), color='blue', label='Google stock price')
plt.plot(train_plot, color='green', label='Training stock price')
plt.plot(test_plot, color='red', label='Predict stock price')
plt.xlabel("Day")
plt.ylabel("Stock Price")
plt.legend()
plt.show()
