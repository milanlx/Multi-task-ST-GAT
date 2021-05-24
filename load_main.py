import numpy as np
from utils.general_utils import loadFromPickle, saveAsPickle
from utils.process_utils import min_max_normalization, initial_df


"""
# Weather
min_dict = {'temp_avg': -5.1, 'rh_avg': 12.0}
max_dict = {'temp_avg': 91.9, 'rh_avg': 100.0}

# normalize weather data
feat_dict = loadFromPickle('feature_data/weather/original/faa.pkl')
years = ['2017', '2018', '2019']
columns = ['temp_avg', 'rh_avg']
for year in years:
    df = feat_dict[year]
    for col in columns:
        min_max_normalization(df, col, min_dict[col], max_dict[col])
    # save
    saveAsPickle(df, 'feature_data/weather/processed/{}.pkl'.format(year))
"""

"""
# Electricity
min_dict = {'ele_increment': 84.0}
max_dict = {'ele_increment': 222.0}

feat_dict = loadFromPickle('feature_data/electricity/original/ele_GHC.pkl')
years = ['2017', '2018', '2019']
columns = ['ele_increment']
building = 'GHC'
# In GHC, outlier in 2018, 2018-03-15 23:15:00 ele_increment is 19
feat_dict['2018'].iat[7065, 1] = 94

for year in years:
    df = feat_dict[year]
    for col in columns:
        min_max_normalization(df, col, min_dict[col], max_dict[col])
    # save
    saveAsPickle(df, 'feature_data/electricity/normalized/{}_{}.pkl'.format(building, year))
"""


# convert to temporal-spatial feature format
prev_step = 3               # number of time steps
feat_dim = 6                # number of feature per node
num_node = 45               # number of node in graph
hours = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
df = loadFromPickle('feature_data/inrix/merged/inrix_merged_feat_df_2017.pkl')
date_time_list = df['date_time'].tolist()
num_rows = len(df.index)
feat_columns = ['st_feature_matrix']
mat_dim = (prev_step*num_node, feat_dim)

imputed_feat = [[0]*feat_dim for _ in range(num_node)]
st_df = initial_df(num_rows, date_time_list, feat_columns)
for i in range(num_rows):
    st_feat = []
    for j in range(i-prev_step, i):
        # zero padding if out-of-range
        if j < 0:
            st_feat.append(imputed_feat)
        else:
            st_feat.append(df.iloc[i,1])
    # convert to numpy array
    st_feat = np.asarray(st_feat)
    st_feat = np.reshape(st_feat, mat_dim)
    st_df.iat[i, 1] = st_feat

print(st_df.loc[2]['st_feature_matrix'])