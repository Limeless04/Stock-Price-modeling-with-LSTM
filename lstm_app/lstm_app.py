import streamlit as st

from datetime import date, datetime
from multiprocessing.spawn import import_main_path
from keras.preprocessing.sequence import TimeseriesGenerator
from pandas_datareader import data, test
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np
from time import time
from sklearn.metrics import r2_score, mean_squared_error
import math

#Variabel
today = date.today()
# print(today)
start_date = '2010-01-01'
end_date = today
look_back = 20

st.title("Real Time Forecasting")

st.sidebar.write("Sidebar")

# masukan = st.sidebar.text_input("Masukan nama saham ? atau")
mn_pilihan = st.sidebar.selectbox("Pilih model saham ? atau",("-","ANTM.JK", "ASII.JK") )
mn_epoch = st.sidebar.select_slider("Berapa Banyak Epoch ?",options=[1, 10, 100])


# mn_look_back = st.sidebar.select_slider("Berapa banyak look back yang diinginkan?", option=[])

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
	dataX, dataY = [], []
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		dataX.append(a)
		dataY.append(dataset[i + look_back, 0])
	return np.array(dataX), np.array(dataY)


#Function  
def main(panel_data, start_date, end_date):
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

    st.write("total time: ", time()-start, 'seconds')
    with st.spinner('Training may take a while, so grab a cup of coffee, or better, go for a run!'):
        model_history = model.fit(trainX, trainY, epochs=mn_epoch, batch_size=1, verbose=1,  validation_split=0.2)
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

    # plot baseline and predictions
    fig2, ax2 = plt.subplots(figsize=(16,9))
    ax2.plot(model_history.history['loss'], label='Loss')
    ax2.plot(model_history.history['val_loss'], label = 'vall_loss') 
    ax2.set_title('model loss')

    ax2.set_xlabel('epoch')
    ax2.set_ylabel('loss')
    ax2.legend()

    st.pyplot(fig2)


if st.sidebar.button('Train'):
    if mn_pilihan == "ANTM.JK":
        pilihan = ["ANTM.JK"]
        lb_saham = "ANTM.JK"
        st.subheader("ANTM Dashboard")
            #Get Data
        panel_data = data.DataReader(pilihan, 'yahoo',start_date, end_date)
        st.write("Data saham " + str(pilihan))
        st.write(panel_data)
        #st.write("Check Missing Value: ", print(close.isnull().sum()))
        st.write("Deskripsi Data " + str(pilihan))
        st.write(panel_data.describe())
        st.write("Starting training with {} epochs...".format(mn_epoch))
        main(panel_data)

    if mn_pilihan == "ASII.JK":
        pilihan = ["ASII.JK"]
        st.subheader("BBNI Dashboard")
        lb_saham = "ASII.JK"
            #Get Data
        panel_data = data.DataReader(pilihan, 'yahoo',start_date, end_date)
        st.write("Data saham " + pilihan)
        st.write(panel_data)
            #st.write("Check Missing Value: ", print(close.isnull().sum()))
        st.write("Deskripsi Data " + pilihan)
        st.write(panel_data.describe())

        main(panel_data)

    # if masukan == "" and pilihan == "-":

    # else:
    #     #st.write(masukan)
    #     lb_saham = masukan
    #     st.subheader("Dashboard saham " + str(lb_saham))
    #     #Get Data
    #     panel_data = data.DataReader(masukan, 'yahoo',start_date, end_date)
    #     st.write("Data saham " + masukan)
    #     st.write(panel_data)
    #     #st.write("Check Missing Value: ", print(close.isnull().sum()))
    #     st.write("Deskripsi Data " + masukan)
    #     st.write(panel_data.describe())

    #     main(panel_data)

