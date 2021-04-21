import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import *


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, npred, nhead, dropout, alpha):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout, alpha, concat=True) for _ in range(nhead)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = GraphAttentionLayer(nhid*nhead, npred, dropout, alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        return x


class STGAT(nn.Module):
    def __init__(self, nfeat, nhid, npred, nhead, dropout, alpha):
        super(STGAT, self).__init__()
        self.dropout = dropout
        self.attentions = [BatchedGraphAttentionLayer(nfeat, nhid, dropout, alpha, concat=True) for _ in range(nhead)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = BatchedGraphAttentionLayer(nhid*nhead, npred, dropout, alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        x = torch.reshape(x, (x.size()[0], x.size()[1] * x.size()[2]))
        return x


class MergeLR(nn.Module):
    def __init__(self, model_gat, in_dim, out_dim):
        super(MergeLR, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.gat = model_gat
        self.linear = nn.Linear(self.in_dim, self.out_dim)
        nn.init.xavier_uniform_(self.linear.weight, gain=1.414)

    def forward(self, x_graph, x_ele, adj):
        x_graph = self.gat(x_graph, adj)
        # concatenate
        x = torch.cat((x_graph, x_ele), dim=1)
        out = self.linear(x)
        return out
