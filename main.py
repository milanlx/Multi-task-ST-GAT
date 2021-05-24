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
npred = 3
dropout = 0.5
alpha = 0.2
nele = 2
adj = np.identity(node)
x = np.random.normal(size=(nsamples, node, nfeat))
y = np.random.normal(size=(nsamples, node*npred))
x_ele = np.random.normal(size=(nsamples, nele))
x, adj = torch.from_numpy(x).float(), torch.from_numpy(adj).float()
x_ele = torch.from_numpy(x_ele).float()
y = torch.from_numpy(y).float()


#model = BatchedGraphAttentionLayer(in_dim=nfeat, out_dim=nhid, dropout=dropout, alpha=alpha)
#output = model(x, adj)

train_loader = DataLoader(STGraphDateset(x, y), batch_size=32, shuffle=True)
model = STGAT(nfeat=nfeat, nhid=nhid, npred=npred, nhead=nhead, dropout=dropout, alpha=alpha)
loss_fun = MultiTaskLoss()
total_model = MergeLR(model_gat=model, in_dim=nele+npred*node, out_dim=npred)
optimizer = optim.Adam(total_model.parameters(), lr=0.01, weight_decay=0.01)
y_hat = total_model(x, x_ele, adj)
print(y_hat.size())

loss_train = []
loss_valid = []
for epoch in range(10):
    temp_loss = 0
    for step, (xx, yy) in enumerate(train_loader):
        model.train()
        output = model(xx, adj)
        loss = loss_fun(output, yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        temp_loss += loss
    loss_train.append(temp_loss)

    # evaluate validation performance
    model.eval()
    with torch.no_grad():
        y_valid = model(x, adj)
        loss = loss_fun(y_valid, y)
        loss_valid.append(loss)


#path = 'stgraph.pkl'
#torch.save(model, path)

for val in loss_valid:
    print(val)




# TODO: batchnorm
