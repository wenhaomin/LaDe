import torch.nn as nn
import torch
import torch.utils.data

class LSTMEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_layers=1, batch_first=True, bidirectional=True):
        super(LSTMEncoder, self).__init__()

        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.num_directions = 2 if self.bidirectional else 1
        self.hidden_size = int(hidden_size / self.num_directions)

        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=self.hidden_size, num_layers=num_layers,
                           batch_first=batch_first, bidirectional=bidirectional)

    def forward(self, embedded_inputs, input_lengths, max_len):
        packed = nn.utils.rnn.pack_padded_sequence(embedded_inputs, torch.tensor([25] * embedded_inputs.shape[0]), batch_first=self.batch_first,
                                                   enforce_sorted=False)

        try:
            outputs, hidden = self.rnn(packed)
        except:
            print('lstm encoder:', embedded_inputs)

        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=self.batch_first)

        extra_padding_size = max_len - outputs.shape[1]
        outputs = nn.functional.pad(outputs, [0, 0, 0, extra_padding_size, 0, 0], mode="constant", value=0)

        if self.bidirectional:
            outputs = torch.cat((outputs[:, :, :self.hidden_size], outputs[:, :, self.hidden_size:]), dim=2)
        batch_size = embedded_inputs.size(0)
        h_n, c_n = hidden
        h_n = h_n.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)
        c_n = c_n.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)
        if self.bidirectional:
            f = (h_n[-1, 0, :, :].squeeze(), c_n[-1, 0, :, :].squeeze())
            b = (h_n[-1, 1, :, :].squeeze(), c_n[-1, 1, :, :].squeeze())
            hidden = (torch.cat((f[0], b[0]), dim=1), torch.cat((f[1], b[1]), dim=1))
        else:
            hidden = (h_n[-1, 0, :, :].squeeze(), c_n[-1, 0, :, :].squeeze())

        return outputs, hidden

class FeaturesLinear(torch.nn.Module):

    def __init__(self, field_dims, output_dim):
        super(FeaturesLinear, self).__init__()
        self.fc = torch.nn.Embedding(field_dims, output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))

    def forward(self, x):

        return torch.sum(self.fc(x), dim=1) + self.bias

class FeaturesEmbedding(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super(FeaturesEmbedding, self).__init__()
        self.embedding = torch.nn.Embedding(field_dims, embed_dim)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        return self.embedding(x)


class FactorizationMachine(torch.nn.Module):

    def __init__(self):
        super(FactorizationMachine, self).__init__()

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        return 0.5 * ix


class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout=0, output_layer=False):
        super(MultiLayerPerceptron, self).__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            if dropout > 0:
                layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        return self.mlp(x.float())


class Regressor(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Regressor, self).__init__()
        self.linear_wide = torch.nn.Linear(input_dim, output_dim, bias=False)
        self.linear_deep = torch.nn.Linear(input_dim, output_dim, bias=False)
        self.linear_recurrent = torch.nn.Linear(input_dim, output_dim, bias=False)
        self.out_layer = MultiLayerPerceptron(output_dim, (output_dim,), output_layer=True)

    def forward(self, wide, deep, recurrent):
        fuse = self.linear_wide(wide) + self.linear_deep(deep) + self.linear_recurrent(recurrent)
        return self.out_layer(fuse)
