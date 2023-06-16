import torch
import torch.nn as nn
import torch.nn.functional as F

class nconv(nn.Module):
    def __init__(self):
        super(nconv, self).__init__()

    def forward(self, x, A):
        # print(x.shape) # [batch_size, n_channels, n_nodes, 23]
        # print(A.shape)
        # input()
        x = torch.einsum('ncvl,vw->ncwl', (x, A))
        return x.contiguous()

class linear(nn.Module):
    def __init__(self, c_in, c_out):
        super(linear, self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(
            1, 1), padding=(0, 0), stride=(1, 1), bias=True)

    def forward(self, x):
        return self.mlp(x)

class GCN(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super(GCN, self).__init__()
        self.nconv = nconv()
        c_in = (order*support_len+1)*c_in
        self.mlp = linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, supports):
        # x: b, c, n, t
        out = [x] # residuals
        for a in supports:
            x1 = self.nconv(x, a)
            # print(f'x1.shape: {x1.shape}')  # [64, 32, 1085, 12]
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                # print(f'x2.shape: {x2.shape}') # [64, 32, 1085, 12]
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=1) # [b, 7c, num_nodes, t]
        # print(h.shape) # [64, 96, 1085, 12]
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h
