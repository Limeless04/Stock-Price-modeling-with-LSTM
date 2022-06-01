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


tickers = ["ANTM.JK","ASII.JK"]

today = date.today()
# print(today)
start_date = '2010-01-01'
end_date = today

panel_data = data.DataReader(tickers, 'yahoo',start_date, end_date)


close = panel_data["Close"]

all_weekdays = pd.date_range(start = start_date,    end= end_date, freq='B')

close = close.reindex(all_weekdays)

close = close.dropna()

#print(all_weekdays)
close.isnull().sum()

close.describe()

# Get the saham timeseries. This now returns a Pandas Series object indexed by date.
lb_saham = 'ANTM.JK'
saham = close.loc[:, lb_saham]


# Calculate the 20 and 100 days moving averages of the closing prices
short_rolling_saham = saham.rolling(window=20).mean()
long_rolling_saham = saham.rolling(window=100).mean()

# Plot everything by leveraging the very powerful matplotlib package
fig, ax = plt.subplots(figsize=(16,9))

ax.plot(saham.index, saham, label=lb_saham)
ax.plot(short_rolling_saham.index, short_rolling_saham, label='20 days rolling')
ax.plot(long_rolling_saham.index, long_rolling_saham, label='100 days rolling')

# Define the date format
date_form = DateFormatter("%b-%y")
ax.xaxis.set_major_formatter(date_form)

ax.set_title('Harga saham {}'.format(lb_saham))
ax.set_xlabel('Date')
ax.set_ylabel('Adjusted closing price ($)')
ax.legend()


train_saham, test_saham = train_test_split(saham.values, test_size=0.2, shuffle=False)


scaler = MinMaxScaler()
train_scale = scaler.fit_transform(train_saham.reshape(-1, 1))
test_scale = scaler.fit_transform(test_saham.reshape(-1, 1))



split=int((1-0.2)*len(saham))

date_train = saham.index[:split]
date_test = saham.index[split:]


look_back = 20
train_gen = TimeseriesGenerator(train_scale, train_scale, length=look_back, batch_size=2)     
test_gen = TimeseriesGenerator(test_scale, test_scale, length=look_back, batch_size=1)



model_forecast = tf.keras.models.Sequential([
  tf.keras.layers.LSTM(32, activation='relu', return_sequences=True, input_shape=(look_back, 1)),
  tf.keras.layers.GlobalMaxPooling1D(),
  tf.keras.layers.Dropout(0.25),
  tf.keras.layers.Dense(1)
])

model_forecast.summary()



from time import time

start = time()



optimizer = tf.keras.optimizers.SGD(learning_rate=1.0000e-04, momentum=0.9)
model_forecast.compile(loss='mse', optimizer='adam')
model_forecast.fit(train_gen, epochs=1, verbose=1)

print("total time: ", time()-start, 'seconds')
pred = scaler.inverse_transform(model_forecast.predict(test_gen))


from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from math import sqrt

mse = round(mean_squared_error(saham.values[split+look_back:],pred), 2)
mae = round(mean_absolute_error(saham.values[split+look_back:],pred), 2)
rmse = round(sqrt(mean_squared_error(saham.values[split+look_back:],pred)),2)
mape = round(mean_absolute_percentage_error(saham.values[split+look_back:],pred),2)
r2 = round(r2_score(saham.values[split+look_back:],pred),2)


# print("MSE score is " + str(mse) + "MAE score is " + str(mae) + "RMSE score is "+ str(rmse) + "MAPE score is "+ str(mape) + "R2 Score is" + str(r2))

scale10 = round((saham.max() - saham.min()) * (10 / 100), 2)

if mae < scale10:
  print("MAE "+str(mae))
else:
  print("MAE " + str(mae) )
  
if mse < scale10:
  print('MSE: ')
else:
  print('MSE: '+str(mse) )

if rmse < scale10:
  print('RMSE: '+str(rmse))
else:
  print('RMSE: '+str(rmse))

print('MAPE: '+ str(mape))

print('R2_score: ' + str(r2))


fig, ax = plt.subplots(figsize=(16,9))

ax.plot(date_train, train_saham, label = "Train data")
ax.plot(date_test[:-look_back], pred, label = "Prediction based in the Test data")
ax.set_title('Price {}'.format(lb_saham))

# Define the date format
date_form = DateFormatter("%b-%y")
ax.xaxis.set_major_formatter(date_form)

ax.set_xlabel('Date',fontsize=15)
ax.set_ylabel('Price',fontsize=15)
ax.legend()
# plt.savefig("bbni_10_pred")
plt.show()


fig, ax = plt.subplots(figsize=(16,9))
#perbandingan data asli dan prediksi
ax.plot(date_test[:-look_back], test_saham.reshape(-1)[:-look_back], label = "Test Data")
ax.plot(date_test[:-look_back], pred, label = "Prediction based in the Test data")

# Define the date format
date_form = DateFormatter("%b-%y")
ax.xaxis.set_major_formatter(date_form)

ax.set_title('Price saham {}'.format(lb_saham))
ax.set_xlabel('Date',fontsize=15)
ax.set_ylabel('Close Price°',fontsize=15)


ax.legend()
# ax.savefig("bbni_10_sidetoside")
plt.show()