import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys
from torch.nn import Sequential, Linear, ReLU

from src.layers.gcn import GCN
from src.base.model import BaseModel


class GWNET(BaseModel):
    def __init__(self,
                 dropout=0.3,
                 supports_len=2,
                 gcn_bool=True,
                 addaptadj=True,
                 aptinit=None,
                 residual_channels=32,
                 dilation_channels=32,
                 skip_channels=256,
                 end_channels=512,
                 kernel_size=2,
                 blocks=4,
                 layers=2,
                 **args):
        super(GWNET, self).__init__(**args)
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers
        self.gcn_bool = gcn_bool
        self.addaptadj = addaptadj

        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.residual_convs = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.gconv = nn.ModuleList()

        self.start_conv = nn.Conv2d(in_channels=self.input_dim,
                                    out_channels=residual_channels,
                                    kernel_size=(1, 1))
        receptive_field = 1

        self.supports_len = supports_len

        if gcn_bool and addaptadj:
            if aptinit is None:
                self.nodevec1 = nn.Parameter(torch.randn(self.num_nodes, 10).to(
                    self.device), requires_grad=True).to(self.device)
                self.nodevec2 = nn.Parameter(torch.randn(10, self.num_nodes).to(
                    self.device), requires_grad=True).to(self.device)
                self.supports_len += 1
            else:
                m, p, n = torch.svd(aptinit)
                initemb1 = torch.mm(m[:, :10], torch.diag(p[:10] ** 0.5))
                initemb2 = torch.mm(torch.diag(p[:10] ** 0.5), n[:, :10].t())
                self.nodevec1 = nn.Parameter(
                    initemb1, requires_grad=True).to(self.device)
                self.nodevec2 = nn.Parameter(
                    initemb2, requires_grad=True).to(self.device)
                self.supports_len += 1

        for b in range(blocks):
            additional_scope = kernel_size - 1
            new_dilation = 1
            for i in range(layers):
                # dilated convolutions
                self.filter_convs.append(nn.Conv2d(in_channels=residual_channels,
                                                   out_channels=dilation_channels,
                                                   kernel_size=(1, kernel_size), dilation=new_dilation,stride=1))

                self.gate_convs.append(nn.Conv1d(in_channels=residual_channels,
                                                 out_channels=dilation_channels,
                                                 kernel_size=(1, kernel_size), dilation=new_dilation,stride=1))

                # 1x1 convolution for residual connection
                self.residual_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                     out_channels=residual_channels,
                                                     kernel_size=(1, 1)))

                # 1x1 convolution for skip connection
                self.skip_convs.append(nn.Conv1d(in_channels=dilation_channels,
                                                 out_channels=skip_channels,
                                                 kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                new_dilation *= 2
                receptive_field += additional_scope
                additional_scope *= 2
                if self.gcn_bool:
                    self.gconv.append(
                        GCN(dilation_channels, residual_channels, dropout, support_len=self.supports_len))

        self.end_conv_1 = nn.Conv2d(in_channels=skip_channels,
                                    out_channels=end_channels,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.end_conv_2 = nn.Conv2d(in_channels=end_channels,
                                    out_channels=self.horizon,
                                    kernel_size=(1, 1),
                                    bias=True)

        self.receptive_field = receptive_field
        if self.seq_len == 24:
            self.mlp_projection = Sequential(Linear(12, 64),
                                    ReLU(),
                                    Linear(64, 128),
                                    ReLU(),
                                    Linear(128, 64),
                                    ReLU(),
                                    Linear(64, self.output_dim)
                                    )

    def forward(self, input, supports):
        # print(f'intput:{input.shape}')

        input = input.permute(0, 3, 2, 1)   #[64, feature_dim, 1085, seq_len]

        # print(f'intput:{input.shape}') 

        in_len = input.size(3)
        if in_len < self.receptive_field:
            x = nn.functional.pad(
                input, (self.receptive_field-in_len, 0, 0, 0))
        else:
            x = input

        x = self.start_conv(x)   #[64, 32, 1085, 24]
        skip = 0

        # print(f'x after start_conv:{x.shape}')

        # calculate the current adaptive adj matrix once per iteration
        new_supports = None
        if self.gcn_bool and self.addaptadj:
            adp = F.softmax(
                F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=1)
            new_supports = supports + [adp]

        # WaveNet layers
        for i in range(self.blocks * self.layers):

            #            |----------------------------------------|     *residual*
            #            |                                        |
            #            |    |-- conv -- tanh --|                |
            # -> dilate -|----|                  * ----|-- 1x1 -- + -->	*input*
            #                 |-- conv -- sigm --|     |
            #                                         1x1
            #                                          |
            # ---------------------------------------> + ------------->	*skip*

            #(dilation, init_dilation) = self.dilations[i]

            #residual = dilation_func(x, dilation, init_dilation, i)


            residual = x    

            # print(f'x_residual:{x.shape}')

            # dilated convolution
            filter = self.filter_convs[i](residual)
            # print(f'filter:{filter.shape}')

            filter = torch.tanh(filter)
            gate = self.gate_convs[i](residual)
            gate = torch.sigmoid(gate)
            # print(gate.shape)
            x = filter * gate

            # print(f'input for GCN? {x.shape}')

            # parametrized skip connection

            s = x
            s = self.skip_convs[i](s)
            try:
                skip = skip[:, :, :,  -s.size(3):]
            except:
                skip = 0
            skip = s + skip

            # print(f'skip_x:{x.shape}') # [64, 32, 1085, 12]

            if self.gcn_bool:
                if self.addaptadj:
                    x = self.gconv[i](x, new_supports)
                else:
                    x = self.gconv[i](x, supports)
            else:
                x = self.residual_convs[i](x)

            # print(f'GCN_x:{x.shape}')

            x = x + residual[:, :, :, -x.size(3):]

            x = self.bn[i](x)

        if self.seq_len == 24:
            # (b, 256, 1085, 12) --> (b, 256, 1085)
            skip = self.mlp_projection(skip)

        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))

        # print(f'end_conv1:{skip.shape}')

        x = self.end_conv_2(x)

        # print(f'output:{x.shape}')  # [64, 12, 1085, 12]

        # b, tc,n_city, n = x.shape
        # return x.reshape(b, tc+n, self.output_dim, n_city).permute(0, 1, 3, 2)
        return x
