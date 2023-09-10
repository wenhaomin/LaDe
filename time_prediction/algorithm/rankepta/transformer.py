# -*- coding: utf-8 -*-
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Attention(nn.Module):
    """A generic attention module for a decoder in seq2seq"""

    def __init__(self, dim, use_tanh=False, C=10):
        super(Attention, self).__init__()
        self.use_tanh = use_tanh
        self.project_query = nn.Linear(dim, dim)
        self.project_ref = nn.Conv1d(dim, dim, 1, 1)
        self.C = C  # tanh exploration
        self.tanh = nn.Tanh()

        self.v = nn.Parameter(torch.FloatTensor(dim))
        self.v.data.uniform_(-(1. / math.sqrt(dim)), 1. / math.sqrt(dim))

    def forward(self, query, ref):
        """
        Args:
            query: is the hidden state of the decoder at the current
                time step. batch x dim
            ref: the set of hidden states from the encoder.
                sourceL x batch x hidden_dim
        """
        # ref is now [batch_size x hidden_dim x sourceL]
        ref = ref.permute(1, 2, 0)
        q = self.project_query(query).unsqueeze(2)  # batch x dim x 1
        e = self.project_ref(ref)  # batch_size x hidden_dim x sourceL
        # expand the query by sourceL
        # batch x dim x sourceL
        expanded_q = q.repeat(1, 1, e.size(2))
        # batch x 1 x hidden_dim
        v_view = self.v.unsqueeze(0).expand(
            expanded_q.size(0), len(self.v)).unsqueeze(1)
        # [batch_size x 1 x hidden_dim] * [batch_size x hidden_dim x sourceL]
        u = torch.bmm(v_view, self.tanh(expanded_q + e)).squeeze(1)
        if self.use_tanh:
            logits = self.C * self.tanh(u)
        else:
            logits = u
        return e, logits

class SkipConnection(nn.Module):

    def __init__(self, module, is_mask = False):
        super(SkipConnection, self).__init__()
        self.module = module
        self.is_mask = is_mask

    # def forward(self, input):
    #     return input + self.module(input)
    #

    def forward(self, input):
        if self.is_mask:
            old_input, h, mask = input
            new_input, h, mask = self.module(input)
            new_input = old_input + new_input
            return (new_input, h, mask)
        else:# 应对 线形层作为 module的时候
            old_input, h, mask = input
            new_input = self.module(old_input)
            new_input = old_input + new_input
            return (new_input, h, mask)

class MultiHeadAttention(nn.Module):
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    # def forward(self, q, h=None, mask=None):
    def forward(self, input):

        """
        :param q: queries (batch_size, n_query, input_dim)
        :param h: data (batch_size, graph_size, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        # print('input: type', type(input))
        # print('input: shape:',len(input))
        q, h, mask = input
        mask = mask.bool()
        # mask = mask.to(torch.bool)
        old_mask =  mask.clone()
        # print('1141, clone')
        if h is None:
            h = q  # compute self-attention

        # h should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = h.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        hflat = h.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(hflat, self.W_key).view(shp)
        V = torch.matmul(hflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))
        if mask is not None:
            # mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility).bool()
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            # mask = mask.to(torch.bool)
            # compatibility[mask] = -np.inf
            # compatibility = torch.Tensor()
            compatibility.masked_fill_(mask, value=-np.inf)

        attn = torch.softmax(compatibility, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc


        # is_vis_attn = False # visualize the attention score
        # if is_vis_attn:
        #     print('vis attention score')
        #     from eva.vis_result import draw_heatmap
        #     attn_vis = attn.clone()
        #     attn_vis = attn_vis.mean(dim=0)
        #     # print('attn SHAPE:', attn_vis.shape)
        #
        #     attn_vis = attn_vis.detach().cpu().numpy()
        #     if not os.path.exists('./attn/'): os.makedirs('./attn/')
        #     for m in attn_vis:
        #         # print(m.shape)
        #         n = np.random.randint(0, 100)
        #         draw_heatmap({'M': m, 'is_int': False, 'fout': f'./attn/{n}_{m.shape[0]}.png'})

        heads = torch.matmul(attn, V)
        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        # return out
        return (out, None, old_mask)

class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

        # Normalization by default initializes affine parameters with bias 0 and weight unif(0,1) which is too large!
        # self.init_parameters()

    def init_parameters(self):

        for name, param in self.named_parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)
    #
    # def forward(self, input):
    #     if isinstance(self.normalizer, nn.BatchNorm1d):
    #         return self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
    #     elif isinstance(self.normalizer, nn.InstanceNorm1d):
    #         return self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
    #     else:
    #         assert self.normalizer is None, "Unknown normalizer type"
    #         return input

    def forward(self, input):
        input, h, mask =  input
        mask = mask.bool()
        if isinstance(self.normalizer, nn.BatchNorm1d):
            input = self.normalizer(input.view(-1, input.size(-1))).view(*input.size())
            return (input,h, mask)

        elif isinstance(self.normalizer, nn.InstanceNorm1d):
            input = self.normalizer(input.permute(0, 2, 1)).permute(0, 2, 1)
            return (input,h, mask)

        else:
            assert self.normalizer is None, "Unknown normalizer type"
            return (input,h, mask)

class MultiHeadAttentionLayer(nn.Sequential):

    def __init__(
            self,
            n_heads,
            embed_dim,
            feed_forward_hidden=512,
            normalization='batch',
    ):
        super(MultiHeadAttentionLayer, self).__init__(
            SkipConnection(
                MultiHeadAttention(
                    n_heads,
                    input_dim=embed_dim,
                    embed_dim=embed_dim
                ),
                is_mask= True,
            ),
            Normalization(embed_dim, normalization),
            SkipConnection(
                nn.Sequential(
                    nn.Linear(embed_dim, feed_forward_hidden),
                    nn.ReLU(),
                    nn.Linear(feed_forward_hidden, embed_dim)
                ) if feed_forward_hidden > 0 else nn.Linear(embed_dim, embed_dim),
                is_mask=False,
            ),
            Normalization(embed_dim, normalization)
        )

class TransformerEncoder(nn.Module):
    def __init__(
            self,
            n_heads,
            embed_dim,
            n_layers,
            node_dim=None,
            normalization='batch',
            feed_forward_hidden=512
    ):
        super(TransformerEncoder, self).__init__()

        # To map input to embedding space
        self.init_embed = nn.Linear(node_dim, embed_dim) if node_dim is not None else None

        self.layers = nn.Sequential(*(
            MultiHeadAttentionLayer(n_heads, embed_dim, feed_forward_hidden, normalization)
            for _ in range(n_layers)
        ))

    def forward(self, x, mask=None):
        # mask implemention
        mask = mask.bool()
        # mask = mask.to(torch.bool)
        h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1) if self.init_embed is not None else x
        h, _, mask = self.layers((h, None, mask))
        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )


def get_transformer_attn_mask(max_seq_len, batch_size, sort_len):
    """
    Get the mask Tensor for Transformer attention
    :return:
    """
    range_tensor = torch.arange(max_seq_len, device=sort_len.device, dtype=sort_len.dtype).expand(batch_size,
                                                                                                  max_seq_len,
                                                                                                  max_seq_len)
    each_len_tensor = sort_len.view(-1, 1, 1).expand(batch_size, max_seq_len, max_seq_len)
    row_mask_tensor = (range_tensor < each_len_tensor)
    col_mask_tensor = row_mask_tensor.transpose(1, 2)
    mask_tensor = row_mask_tensor * col_mask_tensor

    attn_mask = torch.ones((batch_size, max_seq_len, max_seq_len), dtype=torch.long)
    attn_mask[mask_tensor] = 0
    attn_mask = torch.LongTensor(attn_mask)

    return attn_mask
if __name__ == '__main__':
    pass
