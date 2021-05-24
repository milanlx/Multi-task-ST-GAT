import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt


def saveAsPickle(obj, pickle_file):
    with open(pickle_file, 'wb') as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()


def min_max_normalize(df, col_names):
    for col_name in col_names:
        col_max = df[col_name].max()
        col_min = df[col_name].min()
        df[col_name] = df[col_name].apply(lambda x: (x-col_min)/(col_max-col_min))
    return df


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


file_path = '../data/ele_UC.pkl'
with open(file_path, 'rb') as f:
    ele_df = pickle.load(f)

# 2019 only
# columns: [date_time, ele_increment]
col_names = ['ele_increment']
ele_df = min_max_normalize(ele_df, col_names)
data = ele_df['ele_increment'].to_numpy()

# check inf
inf_sum = sum(np.isinf(ele_df['ele_increment']))

feat_df = construct_feature_df(ele_df, 4, 1)


# save
saveAsPickle(feat_df, 'feat_df.pkl')

