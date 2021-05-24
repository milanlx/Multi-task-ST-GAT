from utils.general_utils import loadFromPickle, saveAsPickle
import pandas as pd
import numpy as np


def update_bus_df(df, dirt):
    """convert group_id to 'group_id+dirt' """
    updated_cnt_group = []
    cnt_group = df['connected_group_id'].tolist()
    df['group_id'] = df['group_id'].apply(lambda x: str(x) + '+' + dirt)
    for item in cnt_group:
        if len(item) == 0:
            updated_cnt_group.append(item)
        else:
            updated_item = [str(group_id) + '+' + dirt for group_id in item]
            updated_cnt_group.append(updated_item)
    df['connected_group_id'] = updated_cnt_group
    return df


def get_bus_adj(df):
    """get directed adjacent matrix A[i][j] = 1 if i->j
    :param df: network_df, merged of [dir0, dir1]
    """
    node_list = df['group_id'].tolist()
    neighbors_list = df['connected_group_id'].tolist()
    n = len(node_list)
    adj = np.zeros((n, n))
    for idx, node in enumerate(node_list):
        curr_neighbors = neighbors_list[idx]
        for neighbor in curr_neighbors:
            # can be edges to groups that are filtered out
            if neighbor in node_list:
                adj[idx, node_list.index(neighbor)] = 1
            else:
                print(node_list[idx], neighbor)
    return adj


def reform_bus_feat(feat_dict_dir0, feat_dict_dir1, node_list, feat_columns):
    """
     :param:
         - node_list: ordered sequence of node in graph
         - feat_columns: list of bus feature, [on, off, load]
     :return: df, with columns: [date_time, feature], where feature = [[group_0, group_1,..., group_n]]
     """
    date_time_list = feat_dict_dir0[0]['date_time'].tolist()
    num_rows = len(date_time_list)
    df = initial_df(num_rows, date_time_list, node_list)
    for group_id in node_list:
        idx, dirt = group_id.split('+')
        idx, dirt = int(idx), int(dirt)
        if dirt == 0:
            feat_df = feat_dict_dir0[idx]
        elif dirt == 1:
            feat_df = feat_dict_dir1[idx]
        feat_list = feat_df[feat_columns].values.tolist()
        df[group_id] = feat_list
    # merge
    df['feature_vector'] = df[node_list].values.tolist()
    df.drop(columns=node_list, inplace=True)
    return df


# ------------------------- inrix --------------------------------
def get_inrix_adj(df):
    """ get directed adjacent matrix A[i][j] = 1 if i->j
    :param df: network_df
    """
    neighbors = df['neighbors'].tolist()
    n = len(neighbors)
    adj = np.zeros((n, n))
    for i in range(n):
        curr_neighbors = neighbors[i]
        for neighbor in curr_neighbors:
            adj[i, neighbor] = 1
    return adj


def initial_df(num_rows, date_time_list, feat_columns):
    columns = ['date_time'] + feat_columns
    df = pd.DataFrame(index=np.arange(num_rows), columns=columns)
    df['date_time'] = date_time_list
    return df


def reform_inrix_feat(feat_dict, node_list, feat_columns):
    """
    :param:
        - node_list: ordered sequence of node in graph
        - feat_columns: list of inrix feature, [speed, travel_time]
    :return: df, with columns: [date_time, feature], where feature = [[tmc_0, tmc_1,..., tmc_n]]
    """
    date_time_list = feat_dict[node_list[0]]['date_time'].tolist()
    num_rows = len(date_time_list)
    df = initial_df(num_rows, date_time_list, node_list)
    for tmc_code in node_list:
        feat_df = feat_dict[tmc_code]
        feat_list = feat_df[feat_columns].values.tolist()
        df[tmc_code] = feat_list
    # merge
    df['feature_vector'] = df[node_list].values.tolist()
    df.drop(columns=node_list, inplace=True)
    return df


# BUS
"""
df_dir0 = loadFromPickle('network_data/bus/full_group_df_dir0_filtered.pkl')
df_dir1 = loadFromPickle('network_data/bus/full_group_df_dir1_filtered.pkl')
df_dir0 = update_bus_df(df_dir0, '0')
df_dir1 = update_bus_df(df_dir1, '1')
# merge
df_merged = df_dir0.append(df_dir1)
adj = get_bus_adj(df_merged)
node_list = df_merged['group_id'].tolist()

feat_dict_dir0 = loadFromPickle('feature_data/bus/original/feat_dict_dir0_2019_normalized.pkl')
feat_dict_dir1 = loadFromPickle('feature_data/bus/original/feat_dict_dir1_2019_normalized.pkl')
feat_columns = feat_dict_dir0[0].columns.tolist()[1:]
df = reform_bus_feat(feat_dict_dir0, feat_dict_dir1, node_list, feat_columns)
# save
saveAsPickle(df, 'feature_data/bus/merged/bus_merged_feat_df_2019.pkl')
"""


# INRIX
# inrix, feat_dict columns: [speed_0, speed_1, speed_2, travel_time_0, travel_time_1, travel_time_2]
df = loadFromPickle('network_data/inrix/tmc_df_filter_missing.pkl')
node_list = df['tmc_code'].tolist()
adj = get_inrix_adj(df)

feat_columns = ['speed_0', 'speed_1', 'speed_2', 'travel_time_0', 'travel_time_1', 'travel_time_2']
feat_dict = loadFromPickle('feature_data/inrix/original/inrix_2019_normalized.pkl')

df = reform_inrix_feat(feat_dict, node_list, feat_columns)
# save
saveAsPickle(df, 'feature_data/inrix/merged/inrix_merged_feat_df_2019.pkl')


