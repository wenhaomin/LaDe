# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys, os, math

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.autograd import Variable

import warnings
warnings.filterwarnings("ignore")

from torch.multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass


def masked_log_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    ``torch.nn.functional.log_softmax(vector)`` does not work if some elements of ``vector`` should be
    masked.  This performs a log_softmax on just the non-masked portions of ``vector``.  Passing
    ``None`` in for the mask is also acceptable; you'll just get a regular log_softmax.
    ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
    broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.
    In the case that the input vector is completely masked, the return value of this function is
    arbitrary, but not ``nan``.  You should be masking the result of whatever computation comes out
    of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
    that we deal with this case relies on having single-precision floats; mixing half-precision
    floats with fully-masked vectors will likely give you ``nans``.
    If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
    lower), the way we handle masking here could mess you up.  But if you've got logit values that
    extreme, you've got bigger problems than this.
    """
    if mask is not None:
        # mask = mask.float()
        mask = mask.bool()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.  log(1 + 1e-45) is still basically 0, so we can safely
        # just add 1e-45 before calling mask.log().  We use 1e-45 because 1e-46 is so small it
        # becomes 0 - this is just the smallest value we can actually use.
        vector = vector + (mask + 1e-45).log()
    return torch.nn.functional.log_softmax(vector, dim=dim)


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
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility).bool()
            # compatibility[mask] = -np.inf
            # compatibility = torch.Tensor()
            compatibility.masked_fill_(mask, value=-np.inf)

        attn = torch.softmax(compatibility, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask] = 0
            attn = attnc

        heads = torch.matmul(attn, V)
        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        # return out
        return (out, None, old_mask)

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

    def forward(self, x, mask):

        mask = mask.bool()
        h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1) if self.init_embed is not None else x
        h, _, mask = self.layers((h, None, mask))
        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )


class LSTMEncoder(nn.Module):
    def __init__(self, embedding_dim, hidden_size, num_layers=1, batch_first=True, bidirectional=True):
        super(LSTMEncoder, self).__init__()

        self.batch_first = batch_first
        self.bidirectional = bidirectional
        self.num_layers =  num_layers
        self.embedding_dim =  embedding_dim
        self.num_directions = 2 if self.bidirectional else 1
        self.hidden_size = int(hidden_size / self.num_directions)


        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=self.hidden_size, num_layers=num_layers,
                           batch_first=batch_first, bidirectional=bidirectional)


    def forward(self, embedded_inputs, input_lengths, max_len):
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded_inputs, input_lengths.cpu(), batch_first=self.batch_first,enforce_sorted=False)
        # Forward pass through RNN
        try:
            outputs, hidden = self.rnn(packed)
        except:
            print('lstm encoder:', embedded_inputs)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=self.batch_first)
        # Unpack函数只能padding至当前batch最大长度，需继续pad至全局最大长度
        extra_padding_size = max_len - outputs.shape[1]
        outputs = nn.functional.pad(outputs, [0,0,0,extra_padding_size,0,0], mode="constant", value=0)

        # Return output and final hidden state
        if self.bidirectional:

            outputs = torch.cat((outputs[:, :, :self.hidden_size], outputs[:, :, self.hidden_size:]), dim=2)
        batch_size = embedded_inputs.size(0)
        h_n, c_n = hidden
        h_n = h_n.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)
        c_n = c_n.view(self.num_layers, self.num_directions, batch_size, self.hidden_size)
        if self.bidirectional:
            f = (h_n[-1, 0, :, :].squeeze(), c_n[-1, 0, :, :].squeeze())
            b =  (h_n[-1, 1, :, :].squeeze(), c_n[-1, 1, :, :].squeeze())
            hidden = (torch.cat((f[0], b[0]), dim=1), torch.cat((f[1], b[1]), dim=1))
        else:
            hidden = (h_n[-1, 0, :, :].squeeze(), c_n[-1, 0, :, :].squeeze())


        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 tanh_exploration,
                 use_tanh,
                 n_glimpses=1,
                 mask_glimpses=True,
                 mask_logits=True,
                 geo_vocab_size = 10):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_glimpses = n_glimpses
        self.mask_glimpses = mask_glimpses
        self.mask_logits = mask_logits
        self.use_tanh = use_tanh
        self.tanh_exploration = tanh_exploration
        self.decode_type = 'greedy'  # Needs to be set explicitly before use

        self.lstm = nn.LSTMCell(embedding_dim, hidden_dim)
        self.pointer = Attention(hidden_dim, use_tanh=use_tanh, C=tanh_exploration)
        self.glimpse = Attention(hidden_dim, use_tanh=False)
        self.sm = nn.Softmax(dim=1)

    def check_mask(self, mask_):
        def mask_modify(mask):
            all_true = mask.all(1)  # 一条路线中，不能再继续走了，全为True，此时mask中该路线返回值为True
            mask_mask = torch.zeros_like(mask)  # mask_mask 初始化时全为false
            mask_mask[:,
            -1] = all_true  # 如果该路线走完了，该路线对应的all_true值为true, mask_mask[: , -1]=true 否则为false，mask_mask[: , -1]=flase
            return mask.masked_fill(mask_mask, False)

        return mask_modify(mask_)

    def update_mask(self, mask, selected):
        def mask_modify(mask):
            all_true = mask.all(1)
            mask_mask = torch.zeros_like(mask)
            mask_mask[:, -1] = all_true
            return mask.masked_fill(mask_mask, False)

        result_mask = mask.clone().scatter_(1, selected.unsqueeze(-1), True)
        return mask_modify(result_mask)
        # return mask.clone().scatter_(1, selected.unsqueeze(-1), True)

    def recurrence(self, x, h_in, prev_mask, prev_idxs, step, context):

        logit_mask = self.update_mask(prev_mask, prev_idxs) if prev_idxs is not None else prev_mask
        if prev_idxs == None:  # 如果是第一步
            logit_mask = self.check_mask(logit_mask)

        logits, h_out = self.calc_logits(x, h_in, logit_mask, context, self.mask_glimpses, self.mask_logits)

        # Calculate log_softmax for better numerical stability
        log_p = torch.log_softmax(logits, dim=1)
        probs = log_p.exp()

        if not self.mask_logits:
            # If self.mask_logits, this would be redundant, otherwise we must mask to make sure we don't resample
            # Note that as a result the vector of probs may not sum to one (this is OK for .multinomial sampling)
            # But practically by not masking the logits, a model is learned over all sequences (also infeasible)
            # while only during sampling feasibility is enforced (a.k.a. by setting to 0. here)
            probs[logit_mask] = 0.
            # For consistency we should also mask out in log_p, but the values set to 0 will not be sampled and
            # Therefore not be used by the reinforce estimator

        return h_out, log_p, probs, logit_mask

    def calc_logits(self, x, h_in, logit_mask, context, mask_glimpses=None, mask_logits=None):

        if mask_glimpses is None:
            mask_glimpses = self.mask_glimpses

        if mask_logits is None:
            mask_logits = self.mask_logits
        #print('x shape:', x.shape, 'h_in shape',h_in[0].shape)
        hy, cy = self.lstm(x, h_in)
        g_l, h_out = hy, (hy, cy)

        for i in range(self.n_glimpses):
            ref, logits = self.glimpse(g_l, context)
            # For the glimpses, only mask before softmax so we have always an L1 norm 1 readout vector
            if mask_glimpses:
                logits[logit_mask] = -np.inf
            # [batch_size x h_dim x sourceL] * [batch_size x sourceL x 1] =
            # [batch_size x h_dim x 1]
            g_l = torch.bmm(ref, self.sm(logits).unsqueeze(2)).squeeze(2)
        _, logits = self.pointer(g_l, context)

        # Masking before softmax makes probs sum to one
        if mask_logits:
            logits[logit_mask] = -np.inf

        return logits, h_out

    def forward(self, decoder_input, embedded_inputs, hidden, context, V_reach_mask):
        """
        Args:
            decoder_input: The initial input to the decoder
                size is [batch_size x embedding_dim]. Trainable parameter.
            embedded_inputs: [sourceL x batch_size x embedding_dim]
            hidden: the prev hidden state, size is [batch_size x hidden_dim].
                Initially this is set to (enc_h[-1], enc_c[-1])
            context: encoder outputs, [sourceL x batch_size x hidden_dim]
        """

        batch_size = context.size(1)
        outputs = []
        selections = []
        steps = range(embedded_inputs.size(0))
        idxs = None
        mask = Variable(V_reach_mask, requires_grad=False)

        for i in steps:

            hidden, log_p, probs, mask = self.recurrence(decoder_input, hidden, mask, idxs, i, context)
            # select the next inputs for the decoder [batch_size x hidden_dim]
            idxs = self.decode(
                probs,
                mask
            )

            idxs = idxs.detach()  # Otherwise pytorch complains it want's a reward, todo implement this more properly?

            # Gather input embedding of selected
            decoder_input = torch.gather(
                embedded_inputs,
                0,
                idxs.contiguous().view(1, batch_size, 1).expand(1, batch_size, *embedded_inputs.size()[2:])
            ).squeeze(0)

            # use outs to point to next object
            outputs.append(log_p)
            selections.append(idxs)

        return (torch.stack(outputs, 1), torch.stack(selections, 1))

    def decode(self, probs, mask):
        if self.decode_type == "greedy":
            _, idxs = probs.max(1)
            # assert not mask.gather(1, idxs.unsqueeze(-1)).data.any(), \
            #     "Decode greedy: infeasible action has maximum probability"
        elif self.decode_type == "sampling":
            idxs = probs.multinomial(1).squeeze(1)
            # Check if sampling went OK, can go wrong due to bug on GPU
            while mask.gather(1, idxs.unsqueeze(-1)).data.any():
                print(' [!] resampling due to race condition')
                idxs = probs.multinomial().squeeze(1)
        else:
            assert False, "Unknown decode type"

        return idxs

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

class PointNet(nn.Module):
    def __init__(self, args={}):
        super(PointNet, self).__init__()
        self.args = args

        # network parameters
        self.hidden_size = args['hidden_size']
        self.sort_x_size = args['sort_x_size']

        self.sort_encoder = TransformerEncoder(node_dim=self.hidden_size, embed_dim=self.hidden_size,
                                               n_heads=8, n_layers=2,
                                               normalization='batch')

        # for sort_x embedding layer
        self.sort_x_embedding = nn.Linear(in_features=self.sort_x_size, out_features=self.hidden_size, bias=False)

        tanh_clipping = 10
        mask_inner = True
        mask_logits = True
        self.decoder = Decoder(
            self.hidden_size,#self.sort_x_emb_size
            self.hidden_size,#self.sort_emb_size
            tanh_exploration=tanh_clipping,  # tanh_clipping
            use_tanh=tanh_clipping > 0,
            n_glimpses=1,
            mask_glimpses=mask_inner,
            mask_logits=mask_logits,
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)


    def enc_sort_emb(self, sort_emb, mask_index, batch_size, max_seq_len):
        """
        Encode the sort emb and paper the input for Decoder
        """
        attn_mask = (~mask_index + 0).repeat_interleave(25).reshape(batch_size, max_seq_len, max_seq_len)
        attn_mask = attn_mask.to(sort_emb.device)
        sort_encoder_outputs, emb = self.sort_encoder(sort_emb, attn_mask)
        dec_init_state = (emb, emb)
        decoder_input = sort_encoder_outputs.new_zeros(torch.Size((batch_size, self.hidden_size)))
        inputs = sort_encoder_outputs.permute(1, 0, 2).contiguous()
        enc_h = sort_encoder_outputs.permute(1, 0, 2).contiguous()  #(seq_len, batch_size, hidden)
        return decoder_input, inputs, dec_init_state, enc_h


    def forward(self, V, V_reach_mask):
        B = V_reach_mask.size(0)
        T = V_reach_mask.size(1)
        N = V_reach_mask.size(2)

        mask_index = V.reshape(-1, N, V.shape[-1])[:, :, 0] == 0
        sort_x_emb = self.sort_x_embedding(V.reshape(B*T, N, -1).float())  # (batch_size, max_seq_len, todo_emb_dim)
        decoder_input, inputs, dec_init_state, enc_h = self.enc_sort_emb(sort_x_emb, mask_index, B * T, N)
        (pointer_log_scores, pointer_argmaxs) = self.decoder(decoder_input, inputs, dec_init_state, enc_h, V_reach_mask.reshape(-1, N))
        pointer_log_scores = pointer_log_scores.exp()
        return pointer_log_scores, pointer_argmaxs

    def model_file_name(self):
        file_name = '+'.join([f'{k}${self.args[k]}' for k in [ 'hidden_size', 'dataset']])
        file_name = f'{file_name}.route'
        return file_name

#-------------------------------------------------------------------------------------------------------------------------#
from utils.util import save2file_meta
def save2file(params):
    from utils.util import ws
    file_name = ws + f'/output/route_prediction/{params["dataset"]}/{params["model"]}.csv'
    # 写表头
    head = [
        # data setting
        'dataset', 'min_task_num', 'max_task_num', 'eval_min', 'eval_max',
        # mdoel parameters
        'model', 'hidden_size',
        # training set
        'num_epoch', 'batch_size', 'lr',  'early_stop',  'is_test', 'log_time',
        'lsd', 'lmd', 'krc',  'hr@1', 'hr@2', 'hr@3',
        'ed','acc@1', 'acc@2', 'acc@3'
    ]
    save2file_meta(params,file_name,head)
