import streamlit as st

# Draw a title and some text to the app:
'''
# 
# Prediksi Nilai Return Saham \n
# 
'''

# Load libraries
import numpy as np
import pandas as pd
from itertools import cycle

import pandas_datareader.data as web
# import pandas_datareader.yahoo.daily as yahoo
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
import seaborn as sns
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor

# Libraries for Deep Learning Models
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.layers import LSTM
from keras.wrappers.scikit_learn import KerasRegressor

# Libraries for Statistical Models
import statsmodels.api as sm

# Libraries for Saving the Model
from pickle import dump
from pickle import load

# Time series Models
from statsmodels.tsa.arima.model import ARIMA
# from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Error Metrics
from sklearn.metrics import mean_squared_error

# Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_regression


# Plotting
from pandas.plotting import scatter_matrix
from statsmodels.graphics.tsaplots import plot_acf
import plotly.graph_objects as go 
import plotly.express as px
from plotly.subplots import make_subplots

# Finance library
import yfinance as yf
from datetime import datetime, timedelta
from pandas_datareader.fred import FredReader

# Variabel global
return_period = 5

# Variabel ARIMA
p_values = [0, 1, 2]
d_values = range(0, 2)
q_values = range(0, 2)

# Fungsi
# LSTM Network
def create_LSTMmodel(neurons=12, learn_rate=0.01, momentum=0):
    # create model
    model = Sequential()
    model.add(LSTM(50, input_shape=(
        X_train_LSTM.shape[1], X_train_LSTM.shape[2])))
    #More number of cells can be added if needed
    model.add(Dense(1))
    optimizer = SGD(lr=learn_rate, momentum=momentum)
    model.compile(loss='mse', optimizer="adam")
    return model

# Arima Model 

# warnings.filterwarnings("ignore")
def evaluate_arima_model(arima_order):
    #predicted = list()
    modelARIMA = ARIMA(endog=Y_train, exog=X_train_ARIMA, order=arima_order)
    model_fit = modelARIMA.fit()
    error = mean_squared_error(Y_train, model_fit.fittedvalues)
    return error

def evaluate_models(p_values, d_values, q_values):
    best_score, best_cfg = float("inf"), None
    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    mse = evaluate_arima_model(order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print('ARIMA%s MSE=%.7f' % (order, mse))
                except:
                    continue
    # print('Best ARIMA%s MSE=%.7f' % (best_cfg, best_score))

date = st.date_input(
    "Masukkan tanggal input terakhir")

number = st.number_input('Masukkan jangka waktu (tahunan)')

# get data for the past year
if st.button('Generate'):
    today = date
    start = today - timedelta(days=365.3*number)
    end = today

    stk_data = yf.download(
        tickers="MSFT IBM GOOGL",
        start=start,
        end=end,
    )

    ccy_tickers = ['DEXJPUS', 'DEXUSUK']
    idx_tickers = ['SP500', 'DJIA', 'VIXCLS']

    ccy_data = FredReader(
        symbols=ccy_tickers, 
        start=start, 
        end=end,
        )
    idx_data = FredReader(
        symbols=idx_tickers, 
        start=start, 
        end=end,
        )
    
    st.write('Dataframe harga saham perusahaan Google, IBM, dan Microsoft pada Periode 2012 - 2022')
    st.text(f'periode {start} sampai {end}')
    st.dataframe(stk_data)
    date_column = stk_data.index.to_pydatetime()

    nama = cycle(['Microsoft', 'IBM', 'GOOGLE'])

    fig = px.line(stk_data, x=date_column, y=[stk_data.loc[:, 'Adj Close'].MSFT, stk_data.loc[:, 'Adj Close'].IBM, stk_data.loc[:, 'Adj Close'].GOOGL], labels={'date':'Date', 'close':'Adjusted Close Stock'})
    fig.update_traces(marker_line_width=2)
    fig.update_layout(title_text='Harga Saham Adjusted Close', plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Company List')
    fig.for_each_trace(lambda t: t.update(name=next(nama)))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig)

    st.write('Kita memerlukan nilai tukar mata uang antara negara perusahaan yang bersangkutan, dimana karena perusahaan yang diteliti merupakan merupakan perusahaan asal Amerika Serikat, maka saya mengambil sampel nilai tukar mata uang yen terhadap dollar dan nilai tukar mata uang dollar terhadap poundsterling.')
    st.text(f'periode {start} sampai {end}')
    st.dataframe(ccy_data.read())

    st.write('Kita memerlukan nilai Indeks Harga Saham Gabungan (IHSG), pada penelitian ini saya menggunakan indeks pasar gabungan dari S&P50O, DJIA, dan VIXCLS.')
    st.text(f'periode {start} sampai {end}')
    st.dataframe(idx_data.read())

    st.write('Berikut merupakan perbandingan algoritma terbaik yang akan digunakan untuk menentukan nilai return selama 10 tahun terakhir.')
    st.text(f'periode {start} sampai {end}')

    Y = np.log(stk_data.loc[:, 'Adj Close']['MSFT']).diff(
        return_period).shift(-return_period)
    Y.name = Y.name +'_pred'

    X1 = stk_data.loc[:, 'Adj Close']
    X1.drop(columns=['MSFT'], inplace=True)
    X1 = np.log(X1).diff(return_period)
    X1.columns = X1.columns

    ccy_data = ccy_data.read()
    idx_data = idx_data.read()


    X2 = np.log(ccy_data).diff(return_period)
    X3 = np.log(idx_data).diff(return_period)

    X4 = pd.concat([np.log(stk_data.loc[:, 'Adj Close']['MSFT']) .diff(i) for i in [
               return_period, return_period*3, return_period*6, return_period*12]], axis=1).dropna()
    X4.columns = ['MSFT_DT', 'MSFT_3DT', 'MSFT_6DT', 'MSFT_12DT']

    X = pd.concat([X1, X2, X3, X4], axis=1)

    dataset = pd.concat([Y, X], axis=1).dropna().iloc[::return_period, :]
    Y = dataset.loc[:, Y.name]
    X = dataset.loc[:, X.columns]

    msft_pred = np.array(Y).reshape(-1,1) # variabel msft_pred
    # print(msft_pred.shape)

    st.write('Ini merupakan dataset yang telah melalui tahap preprocessing menggunakan fungsi logaritma untuk menghitung nilai return serta menghilangkan nilai null pada dataset.')
    st.dataframe(dataset)

    validation_size = 0.2
    train_size = int(len(X) * (1-validation_size))
    X_train, X_test = X[0:train_size], X[train_size:len(X)]
    Y_train, Y_test = Y[0:train_size], Y[train_size:len(X)]

    num_folds = 10
    seed = 7
    scoring = 'neg_mean_squared_error'

    models = []
    # models.append(('LR', LinearRegression()))
    # models.append(('LASSO', Lasso()))
    # models.append(('KNN', KNeighborsRegressor()))
    models.append(('SVR_linear', SVR(kernel='linear')))
    models.append(('SVR_poly', SVR(kernel='poly')))
    models.append(('SVR_rbf', SVR(kernel='rbf')))

    # neural network algorithms
    # models.append(('MLP', MLPRegressor()))
    # Boosting methods
    # models.append(('ABR', AdaBoostRegressor()))
    # models.append(('GBR', GradientBoostingRegressor()))
    # Bagging methods
    # models.append(('RFR', RandomForestRegressor()))

    names = []
    kfold_results = []
    test_results = []
    train_results = []
    predict_train = []
    predict_test = []
    for name, model in models:
        names.append(name)

        ## K Fold analysis:

        kfold = KFold(n_splits=num_folds, random_state=seed, shuffle=True)
        #converted mean square error to positive. The lower the beter
        cv_results = -1 * cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
        kfold_results.append(cv_results)

        # Full Training period
        res = model.fit(X_train, Y_train)
        train_predict = res.predict(X_train)
        train_result = mean_squared_error(Y_train, train_predict)
        train_results.append(train_result)

        # predict_train.append(res.predict(X_train))

        # Test results
        test_predict = res.predict(X_test)
        test_result = mean_squared_error(Y_test, test_predict)
        test_results.append(test_result)

        # Plot Visualisasi Arima
        # Using nan to fill column
        trainPredictPlot = np.empty_like(msft_pred)
        trainPredictPlot[:, :] = np.nan
        trainPredictPlot[0:len(train_predict), :] = np.array(train_predict).reshape(-1, 1)
        # print("Train predicted data: ", trainPredictPlot.shape)

        # Shift test predictions for plotting
        testPredictPlot = np.empty_like(msft_pred)
        testPredictPlot[:, :] = np.nan
        testPredictPlot[len(train_predict):len(msft_pred), :] = np.array(test_predict).reshape(-1, 1)
        # print("Test predicted data: ", testPredictPlot.shape)

        date_column = Y.index.to_pydatetime()

        nama = cycle(['Original Stock Return', 'Prediksi Training', 'Prediksi Testing'])
        plotdf = pd.DataFrame({'date': date_column, 'original': Y, 'train_predicted':trainPredictPlot.reshape(1, -1)[0].tolist(), 'test_predicted': testPredictPlot.reshape(1, -1)[0].tolist()})
        # Visualize using plotly
        fig = px.line(plotdf, x=plotdf['date'], y=[plotdf['original'], plotdf['train_predicted'], plotdf['test_predicted']], labels={'value':'Stock price', 'date':'Date'})
        fig.update_layout(title_text=f'Perbandingan Prediksi Stock Return pada {name}', plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
        fig.for_each_trace(lambda t: t.update(name = next(nama)))

        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig)

        msg = "%s: %f (%f) %f %f" % (name, cv_results.mean(),
                                 cv_results.std(), train_result, test_result)
        st.write("-------------------------------------------------------")
        st.write(f"Algoritma = {name}")
        st.write(f"Nilai training Mean Square Error (MSE) : {train_result}")
        st.write(f"Nilai testing Mean Square Error (MSE) : {test_result}")
        st.write("-------------------------------------------------------")
        # st.write(" ")
        # print(msg)



    # Arima model
    X_train_ARIMA = X_train
    X_test_ARIMA = X_test
    tr_len = len(X_train_ARIMA)
    te_len = len(X_test_ARIMA)
    to_len = len(X)
    modelARIMA = ARIMA(Y_train, exog=X_train_ARIMA, order=(1, 0, 0))
    model_fit = modelARIMA.fit()
    train_predict = model_fit.fittedvalues
    error_Training_ARIMA = mean_squared_error(Y_train, train_predict)
    test_predict = model_fit.predict(
        start=tr_len - 1, end=to_len - 1, exog=X_test_ARIMA)[1:]
    error_Test_ARIMA = mean_squared_error(Y_test, test_predict)
    # error_Test_ARIMA

    # st.write(train_predict.shape)

    # Plot Visualisasi Arima
        # Using nan to fill column
    trainPredictPlot = np.empty_like(msft_pred)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[0:len(train_predict), :] = np.array(train_predict).reshape(-1, 1)
    # print("Train predicted data: ", trainPredictPlot.shape)

    # Shift test predictions for plotting
    testPredictPlot = np.empty_like(msft_pred)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict):len(msft_pred), :] = np.array(test_predict).reshape(-1, 1)
    # print("Test predicted data: ", testPredictPlot.shape)

    date_column = Y.index.to_pydatetime()

    nama = cycle(['Original Stock Return', 'Prediksi Training', 'Prediksi Testing'])
    plotdf = pd.DataFrame({'date': date_column, 'original': Y, 'train_predicted':trainPredictPlot.reshape(1, -1)[0].tolist(), 'test_predicted': testPredictPlot.reshape(1, -1)[0].tolist()})
        # Visualize using plotly
    fig = px.line(plotdf, x=plotdf['date'], y=[plotdf['original'], plotdf['train_predicted'], plotdf['test_predicted']], labels={'value':'Stock price', 'date':'Date'})
    fig.update_layout(title_text='Perbandingan Prediksi Stock Return pada ARIMA', plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
    fig.for_each_trace(lambda t: t.update(name = next(nama)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig)

    st.write(f"Mean Square Error Training (ARIMA) : {error_Training_ARIMA}")
    st.write(f"Mean Square Error Testing (ARIMA) : {error_Test_ARIMA}")
   

    # LSTM Model
    seq_len = 2 #Length of the seq for the LSTM

    Y_train_LSTM, Y_test_LSTM = np.array(Y_train)[seq_len-1:], np.array(Y_test)
    X_train_LSTM = np.zeros((X_train.shape[0]+1-seq_len, seq_len, X_train.shape[1]))
    X_test_LSTM = np.zeros((X_test.shape[0], seq_len, X.shape[1]))
    for i in range(seq_len):
        X_train_LSTM[:, i, :] = np.array(X_train)[i:X_train.shape[0]+i+1-seq_len, :]
        X_test_LSTM[:, i, :] = np.array(X)[X_train.shape[0]+i-1:X.shape[0]+i+1-seq_len, :]

    LSTMModel = create_LSTMmodel(12, learn_rate=0.01, momentum=0)
    LSTMModel_fit = LSTMModel.fit(X_train_LSTM, Y_train_LSTM, validation_data=(
        X_test_LSTM, Y_test_LSTM), epochs=300, batch_size=75, verbose=0, shuffle=False)

    # Error LSTM model 
    train_predict = LSTMModel.predict(X_train_LSTM)
    error_Training_LSTM = mean_squared_error(Y_train_LSTM, train_predict)
    test_predict = LSTMModel.predict(X_test_LSTM)
    error_Test_LSTM = mean_squared_error(Y_test, test_predict)

    # Using Nan to fill the column
    # Shift train predictions to plotting
    trainPredictPlot = np.empty_like(msft_pred)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[1:len(train_predict)+1, :] = train_predict
    # print("Train predicted data: ", trainPredictPlot.shape)
        # Shift test predictions for plotting
    testPredictPlot = np.empty_like(msft_pred)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict):len(msft_pred)-1, :] = test_predict
    # print("Test predicted data: ", testPredictPlot.shape)

        # Visualize LSTM using plotly
    nama = cycle(['Original Stock Return', 'Prediksi Training', 'Prediksi Testing'])
    plotdf = pd.DataFrame({'date': date_column, 'original': Y, 'train_predicted':trainPredictPlot.reshape(1, -1)[0].tolist(), 'test_predicted': testPredictPlot.reshape(1, -1)[0].tolist()})

    fig = px.line(plotdf, x=plotdf['date'], y=[plotdf['original'], plotdf['train_predicted'], plotdf['test_predicted']], labels={'value':'Stock price', 'date':'Date'})
    fig.update_layout(title_text='Perbandingan Prediksi Stock Return pada Algoritma LSTM', plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
    fig.for_each_trace(lambda t: t.update(name = next(nama)))

    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig)

    st.write(f"Mean Square Error Training (LSTM) : {error_Training_LSTM}")
    st.write(f"Mean Square Error Testing (LSTM) : {error_Test_LSTM}")

    # Append to previous result
    test_results.append(error_Test_ARIMA)
    test_results.append(error_Test_LSTM)

    train_results.append(error_Training_ARIMA)
    train_results.append(error_Training_LSTM)

    names.append("ARIMA")
    names.append("LSTM")

    # Compare Algorithms
    fig = pyplot.figure()

    ind = np.arange(len(names))  # the x locations 
    width = 0.35  # the width of the bars

    fig.suptitle('Perbandingan Algoritma (Mean Square Error)')
    ax = fig.add_subplot(111)
    pyplot.bar(ind - width/2, train_results,  width=width, label='Train Error')
    pyplot.bar(ind + width/2, test_results, width=width, label='Test Error', hatch='/')
    fig.set_size_inches(15, 8)
    pyplot.legend()
    ax.set_xticks(ind)
    ax.set_xticklabels(names)
    st.pyplot(fig)







