# -*- coding: utf-8 -*-

import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable



class Attention(nn.Module):
    """A generic attention module for a decoder"""

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
        ref = ref.permute(1, 2, 0)  # (64, 512, 25)
        q = self.project_query(query).unsqueeze(2)  # batch x dim x 1, mappign the query into  hidden_dim
        e = self.project_ref(ref)  # batch_size x hidden_dim x sourceL (64,512,25)
        # expand the query by sourceL
        # batch x dim x sourceL
        expanded_q = q.repeat(1, 1, e.size(2))  # (64, 512, 25)
        # batch x 1 x hidden_dim
        v_view = self.v.unsqueeze(0).expand(
            expanded_q.size(0), len(self.v)).unsqueeze(1)  # (64, 1, 512)
        # [batch_size x 1 x hidden_dim] * [batch_size x hidden_dim x sourceL]
        u = torch.bmm(v_view, self.tanh(expanded_q + e)).squeeze(1)  # (B, maxlen)
        if self.use_tanh:
            logits = self.C * self.tanh(u)
        else:
            logits = u
        return e, logits  # e: (64, 512, 25), logits: (64, 25)

class Decoder(nn.Module):
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 worker_fea_dim,
                 tanh_exploration,
                 use_tanh,
                 n_glimpses=1,
                 mask_glimpses=True,
                 mask_logits=True,
                 ):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.worker_fea_dim = worker_fea_dim
        self.n_glimpses = n_glimpses
        self.mask_glimpses = mask_glimpses
        self.mask_logits = mask_logits
        self.use_tanh = use_tanh
        self.tanh_exploration = tanh_exploration
        self.decode_type = 'greedy'  # Needs to be set explicitly before use
        self.merge_linear = nn.Linear(self.hidden_dim + self.worker_fea_dim, self.hidden_dim)
        self.lstm = nn.LSTMCell(embedding_dim, hidden_dim)
        self.pointer = Attention(hidden_dim, use_tanh=use_tanh, C=tanh_exploration)
        self.glimpse = Attention(hidden_dim, use_tanh=False)
        self.sm = nn.Softmax(dim=1)

    def check_mask(self, mask_):
        def mask_modify(mask):
            all_true = mask.all(1)  # all the values of mask should be True when no candidates exists in a route.
            mask_mask = torch.zeros_like(mask)  # mask_mask equal false at first
            mask_mask[:, -1] = all_true # update mask_mask
            return mask.masked_fill(mask_mask, False)
        return mask_modify(mask_)

    def update_mask(self, mask, selected):
        def mask_modify(mask):
            all_true = mask.all(1)
            mask_mask = torch.zeros_like(mask)
            mask_mask[:, -1] = all_true
            return mask.masked_fill(mask_mask, False)

        result_mask = mask.clone().scatter_(1, selected.unsqueeze(-1), True) #mask the selected node, i.e., set the mask of select index to True
        return mask_modify(result_mask)

    def recurrence(self, x, h_in, prev_mask, prev_idxs, step, context, embed_cou, V_decode_mask):
        logit_mask = self.update_mask(prev_mask, prev_idxs) if prev_idxs is not None else prev_mask

        logit_mask = self.check_mask(logit_mask)

        logits, h_out = self.calc_logits(x, h_in, logit_mask, context, embed_cou, V_decode_mask, self.mask_glimpses, self.mask_logits)

        log_p = torch.log_softmax(logits, dim=1)
        probs = log_p.exp()

        if not self.mask_logits:
            probs[logit_mask] = 0.

        return h_out, log_p, probs, logit_mask

    def calc_logits(self, x, h_in, logit_mask, context, embed_cou, V_decode_mask, mask_glimpses=None, mask_logits=None):

        if mask_glimpses is None:
            mask_glimpses = self.mask_glimpses

        if mask_logits is None:
            mask_logits = self.mask_logits
        hy, cy = self.lstm(x, h_in)
        g_l = self.merge_linear(torch.cat([hy, embed_cou], dim=1))
        h_out = (hy, cy)
        for i in range(self.n_glimpses):
            ref, logits = self.glimpse(g_l, context)  # (64, 512, 25), (B, maxlen)

            if mask_glimpses:
                logits[logit_mask] = -np.inf
                logits[V_decode_mask] = -np.inf

            g_l = torch.bmm(ref, self.sm(logits).unsqueeze(2)).squeeze(2)
        _, logits = self.pointer(g_l, context)

        # Masking before softmax makes probs sum to one
        if mask_logits:
            logits[logit_mask] = -np.inf
            logits[V_decode_mask] = -np.inf

        return logits, h_out

    def update_knn_mask(self, idxs, V_decode_mask, D, B, current_node, mask, config):
        """
        :param idxs:
        :param V_decode_mask:
        :param D: distance maxtrix
        :param B:
        :param current_node:
        :param mask:
        :param config:
        :return:
        """

        E_masked_dif = torch.gather(D.permute(1, 0, 2).contiguous(), 0, idxs.view(1, B, 1).expand(1, B, D.size()[2])).squeeze(0) #distance between current node and other available nodes
        valid_sample_index = tuple((torch.sum(((mask == False) + 0),dim=1) > config['k_min_nodes']).nonzero().squeeze(1).tolist())#remain at least k_min_nodes
        if len(valid_sample_index) == 0:
            return V_decode_mask
        else:
            if config['k_nearest_neighbors'] == 'n-1':
                max_dis_idx = tuple(torch.argmax(((E_masked_dif * ((mask == False) + 0))[valid_sample_index, :]).squeeze(1), dim=1).tolist())
                V_decode_mask_ =  V_decode_mask.clone()
                V_decode_mask_[:, current_node + 1, :][valid_sample_index, max_dis_idx] = True
            elif config['k_nearest_neighbors'] == 'n-2':
                max_dis_idx_1 = tuple(
                    ((E_masked_dif * ((mask == False) + 0))[valid_sample_index, :]).topk(2, dim=1)[1][:, 0].tolist())
                max_dis_idx_2 = tuple(
                    ((E_masked_dif * ((mask == False) + 0))[valid_sample_index, :]).topk(2, dim=1)[1][:, 1].tolist())
                V_decode_mask_ = V_decode_mask.clone()
                V_decode_mask_[:, current_node + 1, :][valid_sample_index, max_dis_idx_1] = True
                V_decode_mask_[:, current_node + 1, :][valid_sample_index, max_dis_idx_2] = True
            elif config['k_nearest_neighbors'] == 'n':
                V_decode_mask_ = V_decode_mask.clone()
            else:
                assert False, "Unknown decode type"

        return V_decode_mask_


    def forward(self, decoder_input, embedded_inputs, hidden, context,
                V_reach_mask_t, embed_cou, D, start_idx, config):
        """

        :param decoder_input:
        :param embedded_inputs:
        :param hidden:
        :param context:
        :param V_reach_mask_t:
        :param embed_cou:
        :param D:  distance matix at each time: (B,T,N,N)
        :param config:
        :return:
        """

        B = V_reach_mask_t.size()[0]
        N = V_reach_mask_t.size()[1]
        outputs = []
        selections = []
        steps = range(embedded_inputs.size(0))
        idxs = start_idx
        mask = Variable(V_reach_mask_t, requires_grad=False)  # (B, maxlen)
        V_decode_mask = torch.BoolTensor(np.full([B, N, N], False)).to(decoder_input.device)
        for i in steps:
            V_decode_mask = self.update_knn_mask(idxs, V_decode_mask, D, B, i, mask, config)
            hidden, log_p, probs, mask = self.recurrence(decoder_input, hidden, mask, idxs, i,
                                                         context, embed_cou, V_decode_mask[:, i, :])
            # select the next inputs for the decoder [batch_size x hidden_dim]
            idxs = self.decode(
                probs,
                mask
            )

            # V_decode_mask = self.update_knn_mask(idxs, V_decode_mask, D, B, i, mask, config)
            decoder_input = torch.gather(
                embedded_inputs,  # (maxlen, B, H)
                0,
                idxs.contiguous().view(1, B, 1).expand(1, B, embedded_inputs.size()[2])
            ).squeeze(0)

            # use outs to point to next object
            outputs.append(log_p)
            selections.append(idxs)

        return (torch.stack(outputs, 1), torch.stack(selections, 1), mask)

    def decode(self, probs, mask):
        if self.decode_type == "greedy":
            _, idxs = probs.max(1)
            assert not mask.gather(1, idxs.unsqueeze(-1)).data.any(), \
                "Decode greedy: infeasible action has maximum probability"
        elif self.decode_type == "sampling":
            idxs = probs.multinomial(1).squeeze(1)
            # Check if sampling went OK, can go wrong due to bug on GPU
            while mask.gather(1, idxs.unsqueeze(-1)).data.any():
                print(' [!] resampling due to race condition')
                idxs = probs.multinomial().squeeze(1)
        else:
            assert False, "Unknown decode type"

        return idxs
