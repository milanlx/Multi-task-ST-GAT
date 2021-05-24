import numpy as np
import pandas as pd
import pickle


def loadFromPickle(pickle_file):
    with open(pickle_file, 'rb') as handle:
        unserialized_obj = pickle.load(handle)
    handle.close()
    return unserialized_obj


def construct_feature_df_ele(df, n_prev, n_next):
    """find temporal features and response for electricity data"""
    n = len(df.index)
    columns = ['date_time', 'X', 'Y']
    feat_df = pd.DataFrame(index=np.arange(n), columns=columns)
    date_col, feat_col = df.columns[0], df.columns[1]
    for i in range(n):
        date_time = df.iloc[i][date_col]
        X = df.iloc[max(0,i-n_prev):i][feat_col].to_numpy()
        Y = df.iloc[i:min(i+n_next, n)][feat_col].to_numpy()
        # zero padding. X at head, Y at tail
        if X.shape[0] < n_prev:
            X = np.pad(X, (n_prev-X.shape[0],0), 'constant')
        if Y.shape[0] < n_next:
            Y = np.pad(Y, (0,n_next-Y.shape[0]), 'constant')
        feat_df.loc[i] = [date_time, X, Y]
    return feat_df


def construct_feature_df(df, n_prev):
    """find temporal features for inrix, bus and weather, [n_f * n_p]
    """
    n = len(df.index)
    columns = df.columns
    col_names = ['date_time', 'X']
    feat_df = pd.DataFrame(index=np.arange(n), columns=col_names)
    date_col = df.columns[0]

    for i in range(n):
        date_time = df.iloc[i][date_col]
        curr_row = []

        for feat_col in columns[1:]:
            X = df.iloc[max(0,i-n_prev):i][feat_col].to_numpy()
            # zero padding
            if X.shape[0] < n_prev:
                X = np.pad(X, (n_prev-X.shape[0],0), 'constant')
            curr_row.append(X)
        feat_df.loc[i] = [date_time, np.array(curr_row)]
    return feat_df


# extract features according to time-of-day, day-of-week, range-of-date
def select_df_by_date(feat_df, hours, days, dates):
    """
    :param feat_df: columns [date_time, X, Y]
    :param hours: list of integer, 0-23
    :param days: list of interger, 0-6 (Monday - Sunday)
    :param dates: list of string ('YYYY-mm-dd'), [date_start, date_last]
    :return: df, extracted feat_df
    """
    selected = feat_df.loc[feat_df['date_time'].dt.hour.isin(hours)]
    selected = selected.loc[selected['date_time'].dt.dayofweek.isin(days)]
    mask = (selected['date_time'] >= dates[0]) & (selected['date_time'] <= dates[1])
    selected = selected.loc[mask]
    return selected


# T * n_v * (n_f * n_t)

feat_df = loadFromPickle('feat_df.pkl')
hours = [8,9]
days = [0,1]
dates = ['2019-01-01', '2019-01-10']

feat_df = select_df_by_date(feat_df, hours, days, dates)

n = len(feat_df.index)
feat_df['Y'] = 1
feat_df['X'] = feat_df['Y']
res_df = construct_feature_df(feat_df, 4)
