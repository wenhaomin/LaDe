import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from src.base.model import BaseModel

class TimeBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution to each node of
    a graph in isolation.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param out_channels: Desired number of output channels at each node in
        each time step.
        :param kernel_size: Size of the 1D temporal kernel.
        """
        super(TimeBlock, self).__init__()
        self.c_in = in_channels
        self.c_out = out_channels
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1)
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))
        self.conv2 = nn.Conv2d(in_channels, out_channels, (1, kernel_size))

    def forward(self, x):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels)
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features_out=out_channels)
        """
        # Convert into NCHW format for pytorch to perform convolutions.
        x = x.permute(0, 3, 1, 2)  # [b, c, num_nodes, t]

        
        x_input = self.res_conv(x)
        x_input = x_input[:, :, :, self.kernel_size - 1:]
        out = self.conv1(x) + x_input
        out = out * torch.sigmoid(self.conv2(x))
        # Convert back from NCHW to NHWC
        out = out.permute(0, 2, 3, 1)
        return out


class SpatialBlock(nn.Module):
    def __init__(self, ks, c_in, c_out):
        super(SpatialBlock, self).__init__()
        self.theta = nn.Parameter(torch.FloatTensor(c_in, c_out, ks))
        self.b = nn.Parameter(torch.FloatTensor(1, c_out, 1, 1))
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.theta, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.theta)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.b, -bound, bound)

    def forward(self, x, Lk):
        # x: [b, c_in, time, n_nodes]
        # Lk: [3, n_nodes, n_nodes]
        if len(Lk.shape) == 2: # if supports_len == 1:
            Lk=Lk.unsqueeze(0)
        x_c = torch.einsum("knm,bitm->bitkn", Lk, x)
        x_gc = torch.einsum("iok,bitkn->botn", self.theta,
                            x_c) + self.b  # [b, c_out, time, n_nodes]
        return torch.relu(x_gc + x)


class STGCNBlock(nn.Module):
    """
    Neural network block that applies a temporal convolution on each node in
    isolation, followed by a graph convolution, followed by another temporal
    convolution on each node.
    """

    def __init__(self,
                 in_channels,
                 spatial_channels,
                 out_channels,
                 num_nodes,
                 supports_len):
        """
        :param in_channels: Number of input features at each node in each time
        step.
        :param spatial_channels: Number of output channels of the graph
        convolutional, spatial sub-block.
        :param out_channels: Desired number of output features at each node in
        each time step.
        :param num_nodes: Number of nodes in the graph.
        """
        super(STGCNBlock, self).__init__()
        self.temporal1 = TimeBlock(in_channels=in_channels,
                                   out_channels=out_channels)
        # self.Theta1 = nn.Parameter(torch.FloatTensor(out_channels,
        #  spatial_channels))

        self.spatial = SpatialBlock(supports_len, out_channels, spatial_channels)

        self.temporal2 = TimeBlock(in_channels=spatial_channels,
                                   out_channels=out_channels)
        # self.layer_norm = nn.LayerNorm([num_nodes, out_channels])
        self.layer_norm = nn.LayerNorm([out_channels])
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / math.sqrt(self.Theta1.shape[1])
        # self.Theta1.data.uniform_(-stdv, stdv)
        pass

    def forward(self, X, A_hat):
        """
        :param X: Input data of shape (batch_size, num_nodes, num_timesteps,
        num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        :return: Output data of shape (batch_size, num_nodes,
        num_timesteps_out, num_features=out_channels).
        """
        # print('to temporal1:')
        # print(X.shape)     #[16, 1085, 12, 27] 

        t = self.temporal1(X)  #[16, 1085, 22, 64]
        # print('to spatial:')
        # print(t.shape)

        # # original
        # lfs = torch.einsum("ij,jklm->kilm", [A_hat, t.permute(1, 0, 2, 3)])
        # t2 = F.relu(torch.matmul(lfs, self.Theta1))
        t2 = self.spatial(t.permute(0, 3, 2, 1), A_hat) #[16, 64, 22, 1085]

        # print('to temporal2:')
        # print(t2.shape)

        t3 = self.temporal2(t2.permute(0, 3, 2, 1)) #[16, 1085, 20, 64]

        # print('to layer norm:')
        # print(t3.shape)
        
        # for layer norm
        t3 = t3.permute(0, 2, 1, 3)
        out = self.layer_norm(t3) #[16, 20, 1085, 64]
        # print(out.shape)
        # input()
        return out.permute(0, 2, 1, 3)
        # return t3


class STGCN(BaseModel):
    def __init__(self, n_filters, supports_len, **args):
        super(STGCN, self).__init__(**args)
        self.n_filters = n_filters
        self.supports_len = supports_len
        self.block1 = STGCNBlock(in_channels=self.input_dim, out_channels=self.n_filters,
                                 spatial_channels=self.n_filters, num_nodes=self.num_nodes,supports_len=self.supports_len)
        self.block2 = STGCNBlock(in_channels=self.n_filters, out_channels=self.n_filters,
                                 spatial_channels=self.n_filters, num_nodes=self.num_nodes,supports_len=self.supports_len)
        self.last_temporal = TimeBlock(
            in_channels=self.n_filters, out_channels=self.n_filters)
        self.fully = nn.Linear((self.seq_len - 2 * 5) * self.n_filters,
                               self.horizon*self.output_dim)


    def forward(self, X, supports):
        """
        :param X: Input data of shape (batch_size, num_timesteps, num_nodes, num_features=in_channels).
        :param A_hat: Normalized adjacency matrix.
        """

        # print(X.shape) #[b, l, n, d]

        X = X.permute(0, 2, 1, 3)
        out1 = self.block1(X, supports) #[16, 1085, 20, 64]
        out2 = self.block2(out1, supports) # [16, 1085, 16, 64]
        out3 = self.last_temporal(out2) # [16, 1085, 14, 64]
        out4 = self.fully(out3.reshape((out3.shape[0], out3.shape[1], -1))) #[16, 1085, 24]
        
        # print("out1: {}, out2: {}, out3: {}, out4: {}".format(out1.shape, out2.shape, out3.shape, out4.shape))
        # input()
        b, n, tc = out4.shape
        return out4.reshape(b, n, -1, self.output_dim).permute(0, 2, 1, 3)

        # print('out shape:')
        # print(out4.shape)
        # return out4
