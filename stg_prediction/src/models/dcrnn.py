from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from src.utils import graph_algo
import torch
import torch.nn as nn
from abc import abstractmethod
from src.base.model import BaseModel



class DCRNNModel(BaseModel):
    def __init__(self,
                 max_diffusion_step,
                 num_rnn_layers,
                 n_filters,
                 filter_type,
                 cl_decay_steps,
                 use_curriculum_learning=True,
                 **args):

        super(DCRNNModel, self).__init__(**args)
        self.num_rnn_layers = num_rnn_layers  # should be 2
        self.n_filters = n_filters  # should be 64
        self.encoder = DCRNNEncoder(input_dim=self.input_dim,
                                    max_diffusion_step=max_diffusion_step,
                                    hid_dim=n_filters,
                                    num_nodes=self.num_nodes,
                                    num_rnn_layers=num_rnn_layers,
                                    filter_type=filter_type,
                                    device=self.device)
        self.decoder = DCGRUDecoder(input_dim=self.output_dim,
                                    max_diffusion_step=max_diffusion_step,
                                    num_nodes=self.num_nodes,
                                    hid_dim=n_filters,
                                    output_dim=self.output_dim,
                                    num_rnn_layers=num_rnn_layers,
                                    filter_type=filter_type,
                                    device=self.device)
        assert self.encoder.hid_dim == self.decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"

        self.use_curriculum_learning = use_curriculum_learning
        self.cl_decay_steps = cl_decay_steps
        # self.embedding_air=AirEmbedding()

    def _compute_sampling_threshold(self, batches_seen):
        return self.cl_decay_steps / (
            self.cl_decay_steps + np.exp(batches_seen / self.cl_decay_steps))

    def forward(self, source, target, supports, iter=None):
        # the size of source/target would be (64, 12, 207, 2)

        # print(f"source.shape{source.shape}") #[64, 24, 1085, 16]
        # print(f'target.shape{target.shape}')   #[64, 24, 1085, 1]
        # if self.dataset != 'KnowAir':
        #     x_embed=self.embedding_air(source[...,11:15].long())
        #     source=torch.cat((source[..., :11], x_embed, source[..., 15:]), -1) 

        b, t, n, _ = source.shape
        go_symbol = torch.zeros(1, b, self.num_nodes, self.output_dim).to(self.device)

        source = torch.transpose(source, dim0=0, dim1=1)

        target = torch.transpose(
            target[..., :self.output_dim], dim0=0, dim1=1)
        target = torch.cat([go_symbol, target], dim=0)

        # initialize the hidden state of the encoder
        init_hidden_state = self.encoder.init_hidden(b).to(self.device)

        # last hidden state of the encoder is the context
        # (num_layers, batch, outdim)
        context, _ = self.encoder(source, supports, init_hidden_state)

        if self.training and self.use_curriculum_learning:
            c = np.random.uniform(0, 1)
            teacher_forcing_ratio = self._compute_sampling_threshold(iter)
        else:
            teacher_forcing_ratio = 0

        outputs = self.decoder(
            target, supports, context, teacher_forcing_ratio=teacher_forcing_ratio)
        o = outputs[1:, :, :].permute(1, 0, 2).reshape(b, t, n, self.output_dim)
        # print(f'o.shape{o.shape}') #[64, 24, 1085, 1]
        # the elements of the first time step of the outputs are all zeros.
        return o


class DCRNNEncoder(nn.Module):
    def __init__(self,
                 input_dim,
                 max_diffusion_step,
                 hid_dim,
                 num_nodes,
                 num_rnn_layers,
                 filter_type,
                 device):
        super(DCRNNEncoder, self).__init__()
        self.hid_dim = hid_dim
        self._num_rnn_layers = num_rnn_layers
        self.device = device
        # encoding_cells = []
        encoding_cells = list()
        # the first layer has different input_dim
        encoding_cells.append(DCGRUCell(input_dim=input_dim,
                                        num_units=hid_dim,
                                        max_diffusion_step=max_diffusion_step,
                                        num_nodes=num_nodes,
                                        filter_type=filter_type,
                                        device=device))

        # construct multi-layer rnn
        for _ in range(1, num_rnn_layers):
            encoding_cells.append(DCGRUCell(input_dim=hid_dim,
                                            num_units=hid_dim,
                                            max_diffusion_step=max_diffusion_step,
                                            num_nodes=num_nodes,
                                            filter_type=filter_type,
                                            device=device))
        self.encoding_cells = nn.ModuleList(encoding_cells)

    def forward(self, inputs, supports, initial_hidden_state):
        # inputs shape is (seq_length, batch, num_nodes, input_dim) (12, 64, 207, 2)
        # inputs to cell is (batch, num_nodes * input_dim)
        # init_hidden_state should be (num_layers, batch_size, num_nodes*num_units) (2, 64, 207*64)
        seq_length = inputs.shape[0]
        batch_size = inputs.shape[1]

        # x_embed=self.embedding_air(inputs[...,11:15].long())
        # inputs=torch.cat((inputs[...,:11],x_embed,inputs[...,15:]),-1)  

        inputs = torch.reshape(
            inputs, (seq_length, batch_size, -1))  # (12, 64, 207*2)

        current_inputs = inputs
        # the output hidden states, shape (num_layers, batch, outdim)
        output_hidden = []
        for i_layer in range(self._num_rnn_layers):
            hidden_state = initial_hidden_state[i_layer]
            output_inner = []
            for t in range(seq_length):
                _, hidden_state = self.encoding_cells[i_layer](
                    current_inputs[t, ...], supports, hidden_state)  # (50, 207*64)
                output_inner.append(hidden_state)
            output_hidden.append(hidden_state)
            # seq_len, B, ...
            current_inputs = torch.stack(output_inner, dim=0).to(self.device)
        # output_hidden: the hidden state of each layer at last time step, shape (num_layers, batch, outdim)
        # current_inputs: the hidden state of the top layer (seq_len, B, outdim)
        return output_hidden, current_inputs

    def init_hidden(self, batch_size):
        init_states = []  # this is a list of tuples
        for i in range(self._num_rnn_layers):
            init_states.append(self.encoding_cells[i].init_hidden(batch_size))
        # init_states shape (num_layers, batch_size, num_nodes*num_units)
        return torch.stack(init_states, dim=0)


class DCGRUDecoder(nn.Module):
    def __init__(self,
                 input_dim,
                 max_diffusion_step,
                 num_nodes,
                 hid_dim,
                 output_dim,
                 num_rnn_layers,
                 filter_type,
                 device):
        super(DCGRUDecoder, self).__init__()
        self.hid_dim = hid_dim
        self.num_nodes = num_nodes  # 1085
        self.output_dim = output_dim  # should be 1
        self._num_rnn_layers = num_rnn_layers
        self.device = device
        cell_with_projection = DCGRUCell(input_dim=hid_dim,
                                         num_units=hid_dim,
                                         max_diffusion_step=max_diffusion_step,
                                         num_nodes=num_nodes,
                                         num_proj=output_dim,
                                         filter_type=filter_type,
                                         device=device)

        decoding_cells = list()
        # first layer of the decoder
        decoding_cells.append(DCGRUCell(input_dim=input_dim,
                                        num_units=hid_dim,
                                        max_diffusion_step=max_diffusion_step,
                                        num_nodes=num_nodes,
                                        filter_type=filter_type,
                                        device=device))
        # construct multi-layer rnn
        for _ in range(1, num_rnn_layers - 1):
            decoding_cells.append(DCGRUCell(input_dim=hid_dim,
                                            num_units=hid_dim,
                                            max_diffusion_step=max_diffusion_step,
                                            num_nodes=num_nodes,
                                            filter_type=filter_type,
                                            device=device))
        decoding_cells.append(cell_with_projection)
        self.decoding_cells = nn.ModuleList(decoding_cells)

    def forward(self, inputs, supports, initial_hidden_state, teacher_forcing_ratio=0.5):
        """
        :param inputs: shape should be (seq_length+1, batch_size, num_nodes, input_dim)
        :param initial_hidden_state: the last hidden state of the encoder. (num_layers, batch, outdim)
        :param teacher_forcing_ratio:
        :return: outputs. (seq_length, batch_size, num_nodes*output_dim) (12, 50, 207*1)
        """
        # inputs shape is (seq_length, batch, num_nodes, input_dim) (12, 50, 207, 1)
        # inputs to cell is (batch, num_nodes * input_dim)

        # print('inputs.shape')  # [25, 64, 1085, 1]
        # print(inputs.shape)

        seq_length = inputs.shape[0]  # should be 24+1
        batch_size = inputs.shape[1]  # 64
        inputs = torch.reshape(
            inputs, (seq_length, batch_size, -1))  # (24+1, 64, 1085*1)

        # tensor to store decoder outputs
        outputs = torch.zeros(
            seq_length, batch_size, self.num_nodes*self.output_dim).to(self.device)  # (24+1, 64, 1085*1)
        # if rnn has only one layer
        # if self._num_rnn_layers == 1:
        #     # first input to the decoder is the GO Symbol
        #     current_inputs = inputs[0]  # (64, 207*1)
        #     hidden_state = prev_hidden_state[0]
        #     for t in range(1, seq_length):
        #         output, hidden_state = self.decoding_cells[0](current_inputs, hidden_state)
        #         outputs[t] = output  # (64, 207*1)
        #         teacher_force = random.random() < teacher_forcing_ratio
        #         current_inputs = (inputs[t] if teacher_force else output)

        current_input = inputs[0]  # the first input to the rnn is GO Symbol
        for t in range(1, seq_length):
            # hidden_state = initial_hidden_state[i_layer]  # i_layer=0, 1, ...
            next_input_hidden_state = []
            for i_layer in range(0, self._num_rnn_layers):
                hidden_state = initial_hidden_state[i_layer]
                output, hidden_state = self.decoding_cells[i_layer](
                    current_input, supports, hidden_state)
                current_input = output  # the input of present layer is the output of last layer
                # store each layer's hidden state
                next_input_hidden_state.append(hidden_state)
            initial_hidden_state = torch.stack(next_input_hidden_state, dim=0)
            # store the last layer's output to outputs tensor

            # print(output.shape) # [64, 1085 * 16] 

            outputs[t] = output  # should be [bs, num_cities]
            # perform scheduled sampling teacher forcing
            teacher_force = np.random.random() < teacher_forcing_ratio  # a bool value
            current_input = (inputs[t] if teacher_force else output)

        return outputs


class DiffusionGraphConv(nn.Module):
    def __init__(self,
                 supports_len,
                 input_dim,
                 hid_dim,
                 num_nodes,
                 max_diffusion_step,
                 output_dim,
                 bias_start=0.0):
        super(DiffusionGraphConv, self).__init__()
        # Don't forget to add for x itself.
        self.num_matrices = supports_len * max_diffusion_step + 1
        input_size = input_dim + hid_dim
        self.num_nodes = num_nodes
        self._max_diffusion_step = max_diffusion_step
        self.weight = nn.Parameter(torch.FloatTensor(
            size=(input_size*self.num_matrices, output_dim)))
        self.biases = nn.Parameter(torch.FloatTensor(size=(output_dim,)))
        nn.init.xavier_normal_(self.weight.data, gain=1.414)
        nn.init.constant_(self.biases.data, val=bias_start)

    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 0)
        return torch.cat([x, x_], dim=0)

    def forward(self, inputs, supports, state, output_size, bias_start=0.0):
        """
        Diffusion Graph convolution with graph matrix
        :param inputs:
        :param state:
        :param output_size:
        :param bias_start:
        :return:
        """
        # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
        
        # print(inputs.shape)

        batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self.num_nodes, -1))
        state = torch.reshape(state, (batch_size, self.num_nodes, -1))
        inputs_and_state = torch.cat([inputs, state], dim=2)
        input_size = inputs_and_state.shape[2]
        # dtype = inputs.dtype

        x = inputs_and_state
        x0 = torch.transpose(x, dim0=0, dim1=1)
        # (num_nodes, total_arg_size, batch_size)
        x0 = torch.transpose(x0, dim0=1, dim1=2)
        x0 = torch.reshape(
            x0, shape=[self.num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, dim=0)

        if self._max_diffusion_step == 0:
            pass
        else:
            for support in supports:
                x1 = torch.sparse.mm(support, x0)
                x = self._concat(x, x1)
                for k in range(2, self._max_diffusion_step + 1):
                    x2 = 2 * torch.sparse.mm(support, x1) - x0
                    x = self._concat(x, x2)
                    x1, x0 = x2, x1

        x = torch.reshape(
            x, shape=[self.num_matrices, self.num_nodes, input_size, batch_size])
        # (batch_size, num_nodes, input_size, order)
        x = torch.transpose(x, dim0=0, dim1=3)
        x = torch.reshape(
            x, shape=[batch_size * self.num_nodes, input_size * self.num_matrices])

        # (batch_size * self.num_nodes, output_size)
        x = torch.matmul(x, self.weight)
        x = torch.add(x, self.biases)
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return torch.reshape(x, [batch_size, self.num_nodes * output_size])


class DCGRUCell(nn.Module):
    """
    Graph Convolution Gated Recurrent Unit Cell.
    """
    def __init__(self,
                 input_dim,
                 num_units,
                 max_diffusion_step,
                 num_nodes,
                 num_proj=None,
                 activation=torch.tanh,
                 use_gc_for_ru=True,
                 filter_type='laplacian',
                 device=None):
        """
        :param num_units: the hidden dim of rnn
        :param adj_mat: the (weighted) adjacency matrix of the graph, in numpy ndarray form
        :param max_diffusion_step: the max diffusion step
        :param num_nodes:
        :param num_proj: num of output dim, defaults to 1 (speed)
        :param activation: if None, don't do activation for cell state
        :param use_gc_for_ru: decide whether to use graph convolution inside rnn
        """
        super(DCGRUCell, self).__init__()
        self._activation = activation
        self.num_nodes = num_nodes
        self._num_units = num_units
        self._max_diffusion_step = max_diffusion_step
        self._num_proj = num_proj
        self._use_gc_for_ru = use_gc_for_ru
        self.device = device

        if filter_type == "laplacian":
            supports_len = 1
        elif filter_type == "random_walk":
            supports_len = 1
        elif filter_type == "dual_random_walk":
            supports_len = 2
        else:
            supports_len = 1

        self.dconv_gate = DiffusionGraphConv(supports_len=supports_len,
                                             input_dim=input_dim,
                                             hid_dim=num_units,
                                             num_nodes=num_nodes,
                                             max_diffusion_step=max_diffusion_step,
                                             output_dim=num_units*2)
        self.dconv_candidate = DiffusionGraphConv(supports_len=supports_len,
                                                  input_dim=input_dim,
                                                  hid_dim=num_units, num_nodes=num_nodes,
                                                  max_diffusion_step=max_diffusion_step,
                                                  output_dim=num_units)
        if num_proj is not None:
            self.project = nn.Linear(self._num_units, self._num_proj)

    @property
    def output_size(self):
        output_size = self.num_nodes * self._num_units
        if self._num_proj is not None:
            output_size = self.num_nodes * self._num_proj
        return output_size

    def forward(self, inputs, supports, state):
        """
        :param inputs: (B, num_nodes * input_dim)
        :param state: (B, num_nodes * num_units)
        :return:
        """
        output_size = 2 * self._num_units
        # we start with bias 1.0 to not reset and not update
        if self._use_gc_for_ru:
            fn = self.dconv_gate
        else:
            fn = self._fc
        value = torch.sigmoid(
            fn(inputs, supports, state, output_size, bias_start=1.0))
        value = torch.reshape(value, (-1, self.num_nodes, output_size))
        r, u = torch.split(
            value, split_size_or_sections=int(output_size/2), dim=-1)
        r = torch.reshape(r, (-1, self.num_nodes * self._num_units))
        u = torch.reshape(u, (-1, self.num_nodes * self._num_units))
        # batch_size, self.num_nodes * output_size
        c = self.dconv_candidate(inputs, supports, r * state, self._num_units)
        if self._activation is not None:
            c = self._activation(c)
        output = new_state = u * state + (1 - u) * c
        if self._num_proj is not None:

            # print('self._num_proj')
            # print(self._num_proj)

            # apply linear projection to state
            batch_size = inputs.shape[0]
            # (batch*num_nodes, num_units)
            output = torch.reshape(new_state, shape=(-1, self._num_units))

            # print('output.shape')
            # print(output.shape)
            
            output = torch.reshape(self.project(output), shape=(
                batch_size, self.output_size))  # (50, 207*1)

            # print('output2.shape')
            # print(output.shape)

        return output, new_state

    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 0)
        return torch.cat([x, x_], dim=0)

    @staticmethod
    def _build_sparse_matrix(L):
        """
        build pytorch sparse tensor from scipy sparse matrix
        reference: https://stackoverflow.com/questions/50665141
        :return:
        """
        shape = L.shape
        i = torch.LongTensor(np.vstack((L.row, L.col)).astype(int))
        v = torch.FloatTensor(L.data)
        return torch.sparse.FloatTensor(i, v, torch.Size(shape))

    def _gconv(self, inputs, state, output_size, bias_start=0.0):
        pass

    def _fc(self, inputs, state, output_size, bias_start=0.0):
        pass

    def init_hidden(self, batch_size):
        # state: (B, num_nodes * num_units)
        return torch.zeros(batch_size, self.num_nodes * self._num_units).to(self.device)
