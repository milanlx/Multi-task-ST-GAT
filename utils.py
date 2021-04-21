import numpy as np


def min_max_normalization():
    """normalize features"""
    pass


def z_score_normalization():
    """normalize features"""
    pass


def adj_normalization():
    """normarlization adjacent matrix"""
    pass


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
    # add self-loop
    np.fill_diagonal(adj_st, 1)
    return adj_st




"""
adj = np.zeros((3,3))
adj[0,1] = 1
adj[1,2] = 1
adj[2,0] = 1
nt = 3
adj_st = construct_spatial_temporal_adj(adj, nt)
print(adj_st)
"""
