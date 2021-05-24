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
    def __init__(self, nfeat, nhid, nhead, dropout, alpha):
        super(STGAT, self).__init__()
        self.dropout = dropout
        self.attentions = [BatchedGraphAttentionLayer(nfeat, nhid, dropout, alpha, concat=True) for _ in range(nhead)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)
        self.out_att = BatchedGraphAttentionLayer(nhid*nhead, 1, dropout, alpha, concat=False)

    def forward(self, x, adj):
        x = F.dropout(x, self.dropout, training=self.training)
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.elu(self.out_att(x, adj))
        x = torch.reshape(x, (x.size()[0], x.size()[1] * x.size()[2]))
        return x


class MergeLR(nn.Module):
    def __init__(self, bus_gat, inrix_gat, in_dim, out_dim):
        super(MergeLR, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.bus_gat = bus_gat
        self.inrix_gat = inrix_gat
        self.linear = nn.Linear(self.in_dim, self.out_dim)
        nn.init.xavier_uniform_(self.linear.weight, gain=1.414)

    def forward(self, x_bus_graph, x_inrix_graph, x_ele, x_weather, bus_adj, inrix_adj):
        x_bus_graph = self.bus_gat(x_bus_graph, bus_adj)
        x_inrix_graph = self.inrix_gat(x_inrix_graph, inrix_adj)
        # concatenate
        x = torch.cat((x_bus_graph, x_inrix_graph, x_ele, x_weather), dim=1)
        out = self.linear(x)
        return out


class MultiTaskLossWrapper(nn.Module):
    def __init__(self, ntask, model):
        super(MultiTaskLossWrapper, self).__init__()
        self.model = model
        self.task_num = ntask
        self.log_vars = nn.Parameter(torch.zeros((ntask)))

    def forward(self, x_bus_graph, x_inrix_graph, x_ele, x_weather, bus_adj, inrix_adj, target):
        outputs = self.model(x_bus_graph, x_inrix_graph, x_ele, x_weather, bus_adj, inrix_adj)
        loss = 0
        for i in range(self.task_num):
            precision = torch.exp(-self.log_vars[i])
            curr_loss = torch.sum(precision * (target[i] - outputs[i]) ** 2. + self.log_vars[i], -1)
            loss += curr_loss
        loss = torch.mean(loss)
        return loss, outputs, self.log_vars.data.tolist()
