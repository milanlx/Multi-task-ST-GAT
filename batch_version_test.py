import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


"""test of matmul, softmax"""
# x: batch_size * vertex * features
x = np.zeros((2,3,5))
x[0,:,:] = 1
x[0,1,0] = 2
x[0,2,0] = 3
W = np.ones((5,4))
x_torch = torch.from_numpy(x).float()
W_torch = torch.from_numpy(W).float()

# softmax
#x_sfmx = torch.softmax(x_torch, dim=2)

# matmul
# x_hat = torch.matmul(x_torch, W_torch)

# repeat_interleave
xx_ri = x_torch.repeat_interleave(2, dim=1)

# repeat
xx = x_torch.repeat(1,2,1)
print(xx.squeeze(2))

# a* [xW || xW]
x = np.zeros((2,3,3,4))
x[0,0,0,:] = np.ones(4)
a = np.ones(4)
x_torch = torch.from_numpy(x).float()
a_torch = torch.from_numpy(a).float()
res = torch.matmul(x_torch, a_torch)



