# -*- coding: utf-8 -*-
import numpy as np
import  math
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Categorical
import warnings
warnings.filterwarnings("ignore")

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
            all_true = mask.all(1)
            mask_mask = torch.zeros_like(mask)
            mask_mask[:,
            -1] = all_true
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

    def recurrence(self, x, h_in, prev_mask, prev_idxs, step, context):
        logit_mask = self.update_mask(prev_mask, prev_idxs) if prev_idxs is not None else prev_mask
        if prev_idxs == None:
            logit_mask = self.check_mask(logit_mask)

        logits, h_out = self.calc_logits(x, h_in, logit_mask, context, self.mask_glimpses, self.mask_logits)

        # Calculate log_softmax for better numerical stability
        log_p = torch.log_softmax(logits, dim=1)
        probs = log_p.exp()

        if not self.mask_logits:

            probs[logit_mask] = 0.


        return h_out, log_p, probs, logit_mask

    def calc_logits(self, x, h_in, logit_mask, context, mask_glimpses=None, mask_logits=None):

        if mask_glimpses is None:
            mask_glimpses = self.mask_glimpses

        if mask_logits is None:
            mask_logits = self.mask_logits
        hy, cy = self.lstm(x, h_in)
        g_l, h_out = hy, (hy, cy)

        for i in range(self.n_glimpses):
            ref, logits = self.glimpse(g_l, context) # decoder_input -> decoder hidden state, context
            # For the glimpses, only mask before softmax so we have always an L1 norm 1 readout vector
            if mask_glimpses:
                logits[logit_mask] = -np.inf

            g_l = torch.bmm(ref, self.sm(logits).unsqueeze(2)).squeeze(2)
        _, logits = self.pointer(g_l, context)

        # Masking before softmax makes probs sum to one
        if mask_logits:
            logits[logit_mask] = -np.inf

        return logits, h_out

    def forward(self, decoder_input, embedded_inputs, hidden, context, V_reach_mask, sample, type):
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
        scores = []
        steps = range(embedded_inputs.size(0))
        idxs = None

        mask = Variable(V_reach_mask, requires_grad=False)

        if type == 'mle':
            for i in steps:

                hidden, log_p, probs, mask = self.recurrence(decoder_input, hidden, mask, idxs, i, context) # decoder_input, hidden, context
                # select the next inputs for the decoder [batch_size x hidden_dim]
                idxs, log_prob = self.decode(
                    probs,
                    mask,
                    sample
                )

                idxs = idxs.detach()

                # Gather input embedding of selected
                decoder_input = torch.gather(
                    embedded_inputs,
                    0,
                    idxs.contiguous().view(1, batch_size, 1).expand(1, batch_size, *embedded_inputs.size()[2:])
                ).squeeze(0)

                # use outs to point to next object
                outputs.append(log_p)
                selections.append(idxs) #B * t

            return (torch.stack(outputs, 1), torch.stack(selections, 1),  scores)
        else:
            log_probs = []
            for i in steps:
                hidden, log_p, probs, mask= self.recurrence(decoder_input, hidden, mask, idxs, i, context)
                # select the next inputs for the decoder [batch_size x hidden_dim]
                idxs, log_prob = self.decode(
                    probs,
                    mask,
                    sample
                )

                decoder_input = torch.gather(
                    embedded_inputs,
                    0,
                    idxs.contiguous().view(1, batch_size, 1).expand(1, batch_size, *embedded_inputs.size()[2:])
                ).squeeze(0)

                # use outs to point to next object
                outputs.append(log_p)
                selections.append(idxs)
                log_probs.append(log_prob)

            return (torch.stack(outputs, 1), torch.stack(selections, 1), torch.stack(log_probs, dim=1), scores)

    def decode(self, probs, mask, sample):
        log_prob = torch.tensor([0])
        if sample == False:  # greedy
            _, idxs = probs.max(1)
            assert not mask.gather(1, idxs.unsqueeze(-1)).data.any(), \
                "Decode greedy: infeasible action has maximum probability"
        elif sample == True:
            multi_dist = Categorical(probs)
            idxs = multi_dist.sample()
            log_prob = multi_dist.log_prob(idxs)
            while mask.gather(1, idxs.unsqueeze(-1)).data.any():
                print(' [!] resampling due to race condition')
                idxs = probs.multinomial().squeeze(1)
        else:
            assert False, "Unknown decode type"
        return idxs, log_prob

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
        ref = ref.permute(1, 2, 0).contiguous()
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

    def forward(self, input):
        if self.is_mask:
            old_input, h, mask = input
            new_input, h, mask = self.module(input)
            new_input = old_input + new_input
            return (new_input, h, mask)
        else:
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

    def forward(self, input):
        q, h, mask = input
        mask = mask.bool()
        old_mask =  mask.clone()
        if h is None:
            h = q  # compute self-attention

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

class Normalization(nn.Module):

    def __init__(self, embed_dim, normalization='batch'):
        super(Normalization, self).__init__()

        normalizer_class = {
            'batch': nn.BatchNorm1d,
            'instance': nn.InstanceNorm1d
        }.get(normalization, None)

        self.normalizer = normalizer_class(embed_dim, affine=True)

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

    def forward(self, x, mask):


        # mask implemention
        mask = mask.bool()
        h = self.init_embed(x.view(-1, x.size(-1))).view(*x.size()[:2], -1) if self.init_embed is not None else x
        h, _, mask = self.layers((h, None, mask))
        return (
            h,  # (batch_size, graph_size, embed_dim)
            h.mean(dim=1),  # average to get embedding of graph, (batch_size, embed_dim)
        )

def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis = 1, keepdims = True)
    s = x_exp / x_sum
    return s

#-------------------------------------------------------------------------------------------------------------------------#

class RoutePredictionAgent(nn.Module):
    def __init__(self, args={}):
        super(RoutePredictionAgent, self).__init__()

        # network parameters
        self.hidden_size = args['hidden_size']
        self.sort_x_size = args['sort_x_size']
        self.args = args

        self.n_glimpses = 0
        self.sort_encoder = TransformerEncoder(node_dim=self.hidden_size, embed_dim=self.hidden_size,
                                               n_heads=8, n_layers=2,
                                               normalization='batch')

        ## For sort_x embedding layer
        self.sort_x_embedding = nn.Linear(in_features=self.sort_x_size, out_features=self.hidden_size, bias=False)

        tanh_clipping = 10
        mask_inner = True
        mask_logits = True
        self.decoder = Decoder(
            self.hidden_size,
            self.hidden_size,
            tanh_exploration=tanh_clipping,
            use_tanh=tanh_clipping > 0,
            n_glimpses=1,
            mask_glimpses=mask_inner,
            mask_logits=mask_logits,
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def get_seq_mask(self, max_seq_len, batch_size, sort_len):
        """
        Get the mask Tensor for sort task
        """
        range_tensor = torch.arange(max_seq_len, device=sort_len.device, dtype=sort_len.dtype).expand(batch_size,
                                                                                                      max_seq_len,
                                                                                                      max_seq_len)
        each_len_tensor = sort_len.view(-1, 1, 1).expand(batch_size, max_seq_len, max_seq_len)
        row_mask_tensor = (range_tensor < each_len_tensor)
        col_mask_tensor = row_mask_tensor.transpose(1, 2)
        mask_tensor = row_mask_tensor * col_mask_tensor
        mask_tensor = mask_tensor.bool()
        return mask_tensor

    def get_transformer_attn_mask(self, max_seq_len, batch_size, sort_len):
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

    def get_init_mask(self, max_seq_len, batch_size, sort_len):
        """
        Get the init mask for decoder
        """
        range_tensor = torch.arange(max_seq_len, device=sort_len.device, dtype=sort_len.dtype).expand(batch_size, max_seq_len)
        each_len_tensor = sort_len.view(-1, 1).expand(batch_size, max_seq_len)
        raw_mask_tensor = range_tensor >= each_len_tensor
        return raw_mask_tensor

    def enc_sort_emb(self, sort_emb, batch_size, max_seq_len, mask_index):
        """
        Encode the sort emb and paper the input for Decoder
        """
        attn_mask = (~mask_index +0).repeat_interleave(max_seq_len).reshape(batch_size, max_seq_len, max_seq_len)
        attn_mask = attn_mask.to(sort_emb.device)
        sort_encoder_outputs, emb = self.sort_encoder(sort_emb, attn_mask)  # sort_encoder_outputs:(batch_size, max_seq_len, sort_hidden_size), # emb: (batch_size,  sort_hidden_size)
        dec_init_state = (emb, emb)
        decoder_input = sort_encoder_outputs.new_zeros(torch.Size((batch_size, self.hidden_size)))
        inputs = sort_encoder_outputs.permute(1, 0, 2).contiguous()
        enc_h = sort_encoder_outputs.permute(1, 0, 2).contiguous()  # (seq_len, batch_size, hidden) [B, N, H]
        return decoder_input, inputs, dec_init_state, enc_h

    def forward(self, V, V_reach_mask, sample, type):
        '''
          V: (B, N, 8), the basic feature of a task, incluing lat, lng, area_id, promise_time - query_time, promise - accept_time, relative/absolute distance between target location and current location
          V_reach_mask: (B, T, N), mask made by route constraints
          sample: bool, decode by sample or argmax
          type: mle/rl, decode by MLE of RL
        '''

        B =  V_reach_mask.size(0)
        T =  V_reach_mask.size(1)
        N =  V_reach_mask.size(2)
        mask_index = V.reshape(-1, N, V.shape[-1])[:, :, 0] == 0

        sort_x_emb = self.sort_x_embedding(V.reshape(B * T, N, -1).float())#(B*T, N, H)

        decoder_input, inputs, dec_init_state, enc_h = \
            self.enc_sort_emb(sort_x_emb, B * T, N, mask_index)

        if type == 'mle':
            (pointer_log_scores, pointer_argmaxs, greedy_values) = self.decoder \
                (decoder_input, inputs, dec_init_state, enc_h,  V_reach_mask.reshape(-1, N), sample, type)
            pointer_scores = pointer_log_scores.exp()  # (B * T, N, N)
            return pointer_scores, pointer_argmaxs, greedy_values

        else:
            (pointer_log_scores, pointer_argmaxs, log_probs, values) = self.decoder \
                (decoder_input, inputs, dec_init_state, enc_h, V_reach_mask.reshape(-1, N), sample, type)

            return log_probs, pointer_argmaxs, values

    def model_file_name(self):
        file_name = '+'.join([f'{k}-{self.args[k]}' for k in ['hidden_size']])
        file_name = f'{file_name}.Actor_{time.time()}'
        return file_name

# -------------------------------------------------------------------------------------------------------------------------#
import os


def get_workspace():
    """
    get the workspace path
    :return:
    """
    cur_path = os.path.abspath(__file__)
    file = os.path.dirname(cur_path)
    file = os.path.dirname(file)
    return file


ws = get_workspace()


def dir_check(path):
    """
    check weather the dir of the given path exists, if not, then create it
    """
    import os
    dir = path if os.path.isdir(path) else os.path.split(path)[0]
    if not os.path.exists(dir): os.makedirs(dir)


def save2file_meta(params, file_name, head):
    def timestamp2str(stamp):
        utc_t = int(stamp)
        utc_h = utc_t // 3600
        utc_m = (utc_t // 60) - utc_h * 60
        utc_s = utc_t % 60
        hour = (utc_h + 8) % 24
        t = f'{hour}:{utc_m}:{utc_s}'
        return t

    import csv, time, os
    dir_check(file_name)
    if not os.path.exists(file_name):
        f = open(file_name, "w", newline='\n')
        csv_file = csv.writer(f)
        csv_file.writerow(head)
        f.close()

    with open(file_name, "a", newline='\n') as file:  # linux:\n    windows:\r\n    mac:\r
        csv_file = csv.writer(file)
        params['log_time'] = timestamp2str(time.time())
        data = [params[k] for k in head]
        csv_file.writerow(data)

def save2file(params):
    file_name = ws + f'/output/{params["model"]}.csv'
    head = [
        # data setting
        'dataset', 'min_task_num', 'max_task_num', 'eval_min', 'eval_max', 'state',
        # mdoel parameters
        'model', 'hidden_size', 'rl_ratio', 'trace_decay',
        # training set
        'num_epoch', 'batch_size', 'lr', 'wd', 'early_stop',  'is_test', 'log_time',
        # metric result
        'lsd', 'lmd', 'krc',  'hr@1', 'hr@2', 'hr@3', 'hr@4', 'hr@5', 'hr@6', 'hr@7', 'hr@8', 'hr@9', 'hr@10',
        'ed','acc@1', 'acc@2', 'acc@3', 'acc@4', 'acc@5', 'acc@6', 'acc@7', 'acc@8', 'acc@9', 'acc@10',
    ]
    save2file_meta(params,file_name,head)

