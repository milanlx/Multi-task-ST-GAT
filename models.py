import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer


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

