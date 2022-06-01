import streamlit as st
from datetime import date, datetime
from multiprocessing.spawn import import_main_path
from keras.preprocessing.sequence import TimeseriesGenerator
from pandas_datareader import data
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np
# LSTM for international airline passengers problem with time step regression framing
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from time import time


# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)

def train_st(panel_data, start_date, end_date, mn_epoch, lb_saham):
    #Preperation
    close = panel_data["Close"]
    all_weekdays = pd.date_range(start = start_date,    end= end_date, freq='B')
    close = close.reindex(all_weekdays)
    close = close.dropna()
     # Get the saham timeseries. This now returns a Pandas Series object indexed by date.
    dataframe = close
    dataset = dataframe.values
    dataset = dataset.astype('float32')
    # normalize the dataset
    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(dataset)
    # split into train and test sets
    train_size = int(len(dataset) * 0.67)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

     # reshape into X=t and Y=t+1
    look_back = 2
    trainX, trainY = create_dataset(train, look_back)
    testX, testY = create_dataset(test, look_back)
    # reshape input to be [samples, time steps, features]
    trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
    testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

    start = time()
    # create and fit the LSTM network
    model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(50, activation='tanh', return_sequences=True, input_shape=(look_back, 1)),
    tf.keras.layers.LSTM(50, activation='tanh', return_sequences=True),
    tf.keras.layers.LSTM(50, activation='tanh'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(1)
    ])
    model.summary()
    model.compile(
        loss='mean_absolute_error', 
        optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001),
        metrics = ['mean_absolute_error']
    )

   
    with st.spinner('Training may take a while, so grab a cup of coffee, or better, go for a run!'):
        model_history = model.fit(trainX, trainY, epochs=mn_epoch, batch_size=1, verbose=1,  validation_split=0.2)
        st.write("total time: ", time()-start, 'seconds')
    st.success("Model Trained.")
    # make predictions
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    # invert predictions
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    st.write('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    st.write('Test Score: %.2f RMSE' % (testScore))


    r2TrainScore = r2_score(trainY[0], trainPredict[:,0])
    st.write('Train R2 Score: %.2f R2 Score'% (r2TrainScore))
    r2TestScore = r2_score(testY[0], testPredict[:,0])
    st.write('Train R2 Score: %.2f R2 Score'% (r2TestScore))

    #Date For data
    split = int(len(dataset)* 0.67)
    all_date = dataframe.index
    date_train = close.index[:split]
    date_test = close.index[split:]
    
    # plot baseline and predictions
    fig, ax = plt.subplots(figsize=(16,9))
    ax.plot(all_date,scaler.inverse_transform(dataset), label = "Data Saham")
    ax.plot(date_train[look_back+1:],trainPredict, label = "Prediction based in the Train data")
    ax.plot(date_test[:-look_back-1],testPredict, label = "Prediction based in the Test data")
    ax.set_title('Price {}'.format(lb_saham))

    # Define the date format
    date_form = DateFormatter("%b-%Y")
    ax.xaxis.set_major_formatter(date_form)

    ax.set_xlabel('Date',fontsize=15)
    ax.set_ylabel('Price',fontsize=15)
    ax.legend()
    # plt.savefig("bbni_10_pred")
    st.pyplot(fig)

    # model_history_loss = np.array(model_history.history['loss'])
    # model_history_val_loss = np.array(model_history.history['val_loss'])
    # model_history_loss = pd.DataFrame(model_history.history['loss'])
    # model_history_val_loss = pd.DataFrame(model_history.history['val_loss'])
    # # plot baseline and predictions
    # fig2, ax2 = plt.subplots(figsize=(16,9))
    # ax2.plot(model_history_loss, label='Loss')
    # ax2.plot(model_history_val_loss, label = 'vall_loss') 
    # ax2.set_title('model loss')

    # ax2.set_xlabel('epochs')
    # ax2.set_ylabel('loss')
    # ax2.legend()

    # st.pyplot(fig2)