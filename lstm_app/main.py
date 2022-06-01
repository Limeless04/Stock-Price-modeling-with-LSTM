import streamlit as st
from train_st import *
from datetime import date, datetime

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
        train_st(panel_data, start_date, end_date, mn_epoch, lb_saham)

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
        
        train_st(panel_data, start_date, end_date, mn_epoch, lb_saham)