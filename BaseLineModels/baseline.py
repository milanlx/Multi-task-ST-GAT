import numpy as np
import pickle
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
import matplotlib.pyplot as plt
from pandas import datetime, read_csv
import pandas as pd
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')


def construct_feature_df(df, n_prev, n_next):
    columns = ['date_time', 'X', 'Y']
    feat_df = pd.DataFrame(columns=columns)
    n = len(df.index)
    for i in range(n):
        date_time = df.iloc[i]['date_time']
        X = df.iloc[max(0,i-n_prev):i]['ele_increment'].to_numpy()
        Y = df.iloc[i:min(i+n_next,n)]['ele_increment'].to_numpy()
        # zero padding. X at head, Y at tail
        if X.shape[0] < n_prev:
            X = np.pad(X, (n_prev-X.shape[0],0), 'constant')
        if Y.shape[0] < n_next:
            Y = np.pad(Y, (0,n_next-Y.shape[0]), 'constant')
        # need to revise
        feat_df.loc[len(feat_df)] = [date_time, X, Y]
    return feat_df


def iterative_train(x_series, feat_len, method):
    """Return model
        - method: 'LR', 'RF'
    """
    n = x_series.shape[0]
    min_start = 0
    reg = None
    X = []
    for i in range(n):
        x = x_series[max(min_start, i-feat_len): max(min_start, i)]
        # zero padding
        if x.shape[0] < feat_len:
            x = np.pad(x, (feat_len - x.shape[0], 0), 'constant')
        X.append(x)
    X, Y = np.stack(X, axis=0), x_series
    if method == 'LR':
        reg = LinearRegression().fit(X, Y)
    elif method == 'RF':
        reg = RandomForestRegressor(max_depth=2, random_state=0).fit(X, Y)
    return reg


def iterative_test(reg, x_series, feat_len, lead_time):
    Y_pred = []
    n = x_series.shape[0]
    min_start = 0
    for i in range(n):
        x = x_series[max(min_start, i-feat_len-lead_time+1): max(min_start, i-lead_time+1)]
        # zero padding
        if x.shape[0] < feat_len:
            x = np.pad(x, (feat_len - x.shape[0], 0), 'constant')
        x = x.reshape(1,-1)
        for j in range(lead_time-1):
            x_pred = reg.predict(x)
            x = x[:,1:]
            x = np.append(x, [x_pred], axis=1)
        y_pred = reg.predict(x)
        Y_pred.append(y_pred)
    return np.sqrt(mean_squared_error(Y_pred, x_series)), r2_score(x_series,Y_pred)


def direct_train(x_train, y_train, method):
    """Return model
        - method: 'LR', 'RF'
    """
    reg = None
    if method == 'LR':
        reg = LinearRegression().fit(x_train, y_train)
    elif method == 'RF':
        reg = RandomForestRegressor(max_depth=2, random_state=0).fit(x_train, y_train)
    return reg


def direct_test(x_test, y_test, model):
    y_pred = model.predict(x_test)
    mse = np.sqrt(mean_squared_error(y_pred, y_test))
    r2 = r2_score(y_test, y_pred)
    return mse, r2


def historical_average(x_series, time_windows, n_fold, t_const=96):
    """Historical average
    """
    n = x_series.shape[0]
    mse_error, r2 = [], []
    for window in time_windows:
        X, Y = [], []
        for i in range(n):
            min_start = int(i%t_const)
            x = x_series[max(min_start, i-t_const*(window+1)): max(min_start, i-t_const): t_const]
            # zero padding
            if x.shape[0] < window:
                x = np.pad(x, (window - x.shape[0], 0), 'constant')
            X.append(x)
        # to np.array
        X, Y = np.stack(X, axis=0), x_series
        Y_pred = np.mean(X, axis=1)
        mse = np.sqrt(mean_squared_error(Y, Y_pred))
        mse_error.append(mse)
        r2.append(r2_score(Y, Y_pred))
    return mse_error, r2


def generate_lead_time_data(x_series, feat_len, lead_time):
    """Generate X and Y given lead time for direct approaches
        - lead_time >= 1
    """
    n = x_series.shape[0]
    min_start = 0
    X = []
    for i in range(n):
        x = x_series[max(min_start, i-feat_len-lead_time+1): max(min_start, i-lead_time+1)]
        # zero padding
        if x.shape[0] < feat_len:
            x = np.pad(x, (feat_len - x.shape[0], 0), 'constant')
        X.append(x)
    X, Y = np.stack(X, axis=0), x_series
    return X, Y



def loadFromPickle(pickle_file):
    with open(pickle_file, 'rb') as handle:
        unserialized_obj = pickle.load(handle)
    handle.close()
    return unserialized_obj


feat_df = loadFromPickle('feat_df.pkl')
x = np.stack(feat_df['X'], axis=0)
y = feat_df['Y']


#plot_pacf(y, lags=96)
#plot_acf(y, lags=96*7, use_vlines=False, fft=True, markersize=0.5)
#plt.show()


"""TODOLIST: 
    - arima model, understand pacf/acf
        https://towardsdatascience.com/time-series-forecasting-with-sarima-in-python-cda5b793977b
    - recursive models: 
"""

"""
x_series = np.concatenate(y, axis=0)

#mse_errors = historical_average(x_series, [1,2,3,4,5,6,7,8], 2)

lead_time = 16
X, Y = generate_lead_time_data(x_series, 96, lead_time)
train_idx = int(x_series.shape[0]/2)
x_train, y_train = X[0:train_idx], Y[0:train_idx]
x_test, y_test = X[train_idx::], Y[train_idx::]
"""

"""
# random forest
reg = RandomForestRegressor(max_depth=2, random_state=0).fit(x_train, y_train)
print('regression score: ' + str(reg.score(x_train, y_train)))
y_pred = reg.predict(x_test)
print('MSE: ' + str(np.sqrt(mean_squared_error(y_test, y_pred))))
"""

"""
# regression 
reg = LinearRegression().fit(x_train, y_train)
print('regression score: ' + str(reg.score(x_train, y_train)))
y_pred = reg.predict(x_test)
print('MSE: ' + str(np.sqrt(mean_squared_error(y_test, y_pred))))
"""

"""
x_series = np.concatenate(y, axis=0)
train_idx = int(x_series.shape[0]/2)
feat_len = 96
method = 'RF'
lead_time = 1
x_series_train = x_series[0:train_idx]
x_series_test = x_series[train_idx::]
reg = iterative_train(x_series_train, feat_len, method)
mse, r2 = iterative_test(reg, x_series_test, feat_len, lead_time)
print('MSE: ' + str(mse))
print('R2: ' + str(r2))
"""

#y = y.diff(periods=96)
#y = y.dropna()
x_series = np.concatenate(y.to_numpy(), axis=0)[0:10000]

series = read_csv('../data/shampoo.csv', header=0, index_col=0, parse_dates=True, squeeze=True, date_parser=parser)
series.index = series.index.to_period('M')

#model = SARIMAX(x_series, order=(2,0,0), seasonal_order=(2,0,0,96)).fit()
model = ARIMA(x_series, order=(5,0,0)).fit()
print(model.summary())



