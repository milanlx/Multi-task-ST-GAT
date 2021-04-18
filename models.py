import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer, LinearRegressionLayer


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
    def __init__(self, nfeat, nhid, dropout, alpha):
        super(STGAT, self).__init__()
        self.dropout = dropout
        self.att1 = GraphAttentionLayer(nfeat, nhid, dropout, alpha)
        self.att2 = GraphAttentionLayer(nhid, nhid, dropout, alpha)
        self.lr = LinearRegressionLayer(nhid, int(nhid/2))

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.att1(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.att2(x, adj)
        x = F.dropout(x, self.dropout, training=self.training)
        out = self.lr(x)
        return out

