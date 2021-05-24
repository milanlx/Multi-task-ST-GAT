import pandas as pd
import numpy as np


def initial_df(num_rows, date_time_list, feat_columns):
    """initialize df with fixed number of rows and predefined columns"""
    columns = ['date_time'] + feat_columns
    df = pd.DataFrame(index=np.arange(num_rows), columns=columns)
    df['date_time'] = date_time_list
    return df


def min_max_normalization(df, col_name, min_val, max_val):
    """column-wise normalization, modified in place"""
    df[col_name] = df[col_name].apply(lambda x: (x-min_val)/(max_val-min_val))
    return 0


# adj[i,j] = 1 -> flow from i->j, only 0,1 considering connectivity, self-loop not considered
def construct_spatial_temporal_adj(adj, nt):
    """
    :param adj: directed adjacent matrix at single timestamp, dim: n*n
    :param nt: number of timestamp
    :return: adj_st, dim: (n*nt)*(n*nt)
    """
    n = adj.shape[0]
    adj_st = np.zeros((n*nt, n*nt))
    # fill spatial
    for i in range(nt):
        adj_st[nt*i:nt*(i+1),nt*i:nt*(i+1)] = adj
    # fill temporal, self-loop
    for i in range(nt-1):
        for j in range(n):
            adj_st[i*nt+j, (i+1)*nt+j] = 1
    # fill spatial-temporal
    for i in range(n):
        for j in range(n):
            if adj[i, j] == 1:
                for k in range(nt-1):
                    adj_st[i+k*n, j+(k+1)*n] = 1
    return adj_st


def add_self_loop(adj):
    np.fill_diagonal(adj, 1)
