"""
debug
"""
import torch
import numpy as np
from models import *
from layers import *
import torch.optim as optim
from torch.autograd import Variable
from batchGenerator import *


nsamples = 128
node = 5
nfeat = 4
nhid = 2
nhead = 7
npred = 2
nele = 2
nweather = 3
dropout = 0.8
alpha = 0.2
adj = np.identity(node)
x_bus = np.random.normal(size=(nsamples, node, nfeat))
x_inrix = np.random.normal(size=(nsamples, node, nfeat))
x_ele = np.random.normal(size=(nsamples, nele))
x_weather = np.random.normal(size=(nsamples, nweather))
y = np.random.normal(size=(nsamples, npred))
# convert to torch tensor
x_bus, x_inrix = torch.from_numpy(x_bus).float(), torch.from_numpy(x_inrix).float()
x_ele, x_weather = torch.from_numpy(x_ele).float(), torch.from_numpy(x_weather).float()
adj = torch.from_numpy(adj).float()
y = torch.from_numpy(y).float()

# train
train_loader = DataLoader(TotalSTGraphDateset(x_bus, x_inrix, x_ele, x_weather, y), batch_size=32, shuffle=True)
bus_gat = STGAT(nfeat=nfeat, nhid=nhid, nhead=nhead, dropout=dropout, alpha=alpha)
inrix_gat = STGAT(nfeat=nfeat, nhid=nhid, nhead=nhead, dropout=dropout, alpha=alpha)
total_model = MergeLR(bus_gat=bus_gat, inrix_gat=inrix_gat, in_dim=nele+nweather+node+node, out_dim=npred)
loss_fun = MultiTaskLoss()
optimizer = optim.Adam(total_model.parameters(), lr=0.0001, weight_decay=0.01)

# forward pass
y_hat = total_model(x_bus, x_inrix, x_ele, x_weather, adj, adj)
print(y_hat.size())

loss_train = []
loss_valid = []
for epoch in range(10):
    temp_loss = 0
    for step, (xx_bus, xx_inrix, xx_ele, xx_weather, yy) in enumerate(train_loader):
        total_model.train()
        output = total_model(xx_bus, xx_inrix, xx_ele, xx_weather, adj, adj)
        loss = loss_fun(output, yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        temp_loss += loss
    loss_train.append(temp_loss)
    print(temp_loss.item())
