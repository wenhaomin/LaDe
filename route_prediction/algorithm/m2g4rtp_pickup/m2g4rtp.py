# -*- coding: utf-8 -*-
import numpy as np
import  math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch import Tensor
from torch.distributions import Categorical
import warnings
warnings.filterwarnings("ignore")
import algorithm.m2g4rtp_pickup.gat_encoder as gat_encoder
import algorithm.m2g4rtp_pickup.pointer_decoder as pointer_decoder

class Decoder(nn.Module):
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 tanh_exploration,
                 use_tanh,
                 n_glimpses=1,
                 mask_glimpses=True,
                 mask_logits=True,
                 start_fea = 5):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_glimpses = n_glimpses
        self.mask_glimpses = mask_glimpses
        self.mask_logits = mask_logits
        self.use_tanh = use_tanh
        self.tanh_exploration = tanh_exploration
        self.decode_type = 'greedy'

        self.lstm = nn.LSTMCell(embedding_dim, hidden_dim)
        self.pointer = Attention(hidden_dim, use_tanh=use_tanh, C=tanh_exploration)
        self.glimpse = Attention(hidden_dim, use_tanh=False)
        self.sm = nn.Softmax(dim=1)
        self.first_node_embed = nn.Linear(in_features=start_fea, out_features=hidden_dim, bias=False)
        self.lstm_eta = nn.LSTMCell(self.embedding_dim, hidden_dim)
        self.eta_linear = nn.Linear(in_features=hidden_dim * 2, out_features=1)

    def check_mask(self, mask_):
        def mask_modify(mask):
            all_true = mask.all(1)
            mask_mask = torch.zeros_like(mask)
            mask_mask[:,-1] = all_true
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
            ref, logits = self.glimpse(g_l, context)
            if mask_glimpses:
                logits[logit_mask] = -np.inf

            g_l = torch.bmm(ref, self.sm(logits).unsqueeze(2)).squeeze(2)
        _, logits = self.pointer(g_l, context)

        if mask_logits:
            logits[logit_mask] = -np.inf

        return logits, h_out

    def recurrence_eta(self, h_in, next_node_idxs, context, last_node):
        B = next_node_idxs.shape[0]
        current_node_emb = torch.gather(context, 0, next_node_idxs.view(1, B, 1).expand(1, B, context.shape[2])).squeeze(0)
        current_node_input = current_node_emb
        hy, cy = self.lstm_eta(current_node_input.float(), h_in)
        g_l, h_out = hy, (hy, cy)
        route_state = torch.cat([last_node, current_node_input.float()], dim=1)
        eta = self.eta_linear(route_state)
        return h_out, eta, current_node_input

    def forward(self, start_fea, decoder_input, embedded_inputs, hidden, context, V_reach_mask, V):

        batch_size = context.size(1)
        outputs = []
        selections = []
        eta_prediction = []
        steps = range(embedded_inputs.size(0))
        idxs = None
        mask = Variable(V_reach_mask, requires_grad=False)
        first_node_embed = self.first_node_embed(start_fea.float())
        first_node_input = first_node_embed
        hidden_eta = (hidden[0].clone(), hidden[1].clone())
        hy, cy = self.lstm_eta(first_node_input.float(), hidden_eta)
        hidden_eta = (hy, cy)
        current_eta = 0
        last_node = first_node_input.float()

        for i in steps:
            V_masked = V.reshape(V.shape[0], V.shape[1], V.shape[2]) * (~mask + 0).unsqueeze(2)
            hidden, log_p, probs, mask= self.recurrence( decoder_input, hidden, mask, idxs, i, context)
            idxs, log_prob = self.decode(
                probs,
                mask
            )
            hidden_eta, eta_duration_pred, last_node = self.recurrence_eta(hidden_eta, idxs, context, last_node)
            current_eta = current_eta + eta_duration_pred
            decoder_input = torch.gather(
                embedded_inputs,
                0,
                idxs.contiguous().view(1, batch_size, 1).expand(1, batch_size, *embedded_inputs.size()[2:])
            ).squeeze(0)
            outputs.append(log_p)
            selections.append(idxs)
            eta_prediction.append(current_eta)

        return (torch.stack(outputs, 1), torch.stack(selections, 1), torch.stack(eta_prediction, 1).squeeze(-1))

    def decode(self, probs, mask):
        log_prob = torch.tensor([0])
        _, idxs = probs.max(1)
        assert not mask.gather(1, idxs.unsqueeze(-1)).data.any(), \
            "Decode greedy: infeasible action has maximum probability"

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

def get_init_mask(max_seq_len, batch_size, sort_len):
    """
    Get the init mask for decoder
    """
    range_tensor = torch.arange(max_seq_len, device=sort_len.device, dtype=sort_len.dtype).expand(batch_size, max_seq_len)
    each_len_tensor = sort_len.view(-1, 1).expand(batch_size, max_seq_len)
    raw_mask_tensor = range_tensor >= each_len_tensor
    return raw_mask_tensor

class PN_decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, max_seq_len):
        super(PN_decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.max_seq_len = max_seq_len
        self.start_fea_dim = 4

        self.first_input_layer = nn.Linear(self.start_fea_dim, embedding_dim)
        self.decoder = pointer_decoder.Decoder(embedding_dim,
                                               hidden_dim,
                                               tanh_exploration=10,
                                               use_tanh=True,
                                               n_glimpses=1,
                                               mask_glimpses=True,
                                               mask_logits=True)

    def forward(self, hidden_state, start_fea, seq_len):
        batch = hidden_state.shape[0]

        h = torch.zeros([batch, self.hidden_dim], device=hidden_state.device)
        c = torch.zeros([batch, self.hidden_dim], device=hidden_state.device)
        init_input = self.first_input_layer(start_fea)

        init_mask = get_init_mask(self.max_seq_len, batch, seq_len)
        hidden_state = hidden_state.permute(1, 0, 2)
        score, arg = self.decoder(decoder_input=init_input,
                                  embedded_inputs=hidden_state,
                                  hidden=(h, c),
                                  context=hidden_state,
                                  init_mask=init_mask)
        score = score.exp()
        return score, arg
#-------------------------------------------------------------------------------------------------------------------------#

class M2G4RTP(nn.Module):
    def __init__(self, args={}):
        super(M2G4RTP, self).__init__()

        # network parameters
        self.hidden_size = args['hidden_size']
        self.sort_x_size = args['sort_x_size']
        self.device = args['device']
        self.d_e = 4
        self.args = args
        self.edge_emb = nn.Linear(self.d_e, self.hidden_size, bias=False)
        self.max_seq_len = args['max_task_num']
        self.gat_layers = 2
        self.gat_nhead = 8
        self.aoi_size = 6
        self.aoi_edge_size = 1
        self.start_fea = args['start_fea']
        self.pe_size = self.hidden_size
        self.max_aoi_len = 10

        self.n_glimpses = 0
        self.order_decoder_aoi = PN_decoder(self.hidden_size,
                                            self.hidden_size,
                                            max_seq_len=self.max_aoi_len)

        self.unpick_encoder = gat_encoder.GAT_encoder(node_size=self.hidden_size,
                                                edge_size=self.hidden_size,
                                                hidden_size=self.hidden_size,
                                                num_layers=self.gat_layers,
                                                nheads=self.gat_nhead,
                                                is_mix_attention=True,
                                                is_update_edge=True,
                                                num_node=self.max_seq_len)

        self.input_layer_aoi = nn.Linear(self.aoi_size, self.hidden_size)
        self.input_layer_aoi_edge = nn.Linear(self.aoi_edge_size, self.hidden_size)

        self.sort_x_embedding = nn.Linear(in_features=self.sort_x_size, out_features=self.hidden_size, bias=False)
        self.start_embed = nn.Linear(in_features=self.start_fea, out_features=self.hidden_size + self.sort_x_size + self.pe_size, bias=False)
        self.pe_tabel = self.positional_encoding(self.pe_size).to(self.device)

        tanh_clipping = 10
        mask_inner = True
        mask_logits = True
        self.decoder = Decoder(
            self.hidden_size + self.sort_x_size + self.pe_size,
            self.hidden_size + self.sort_x_size + self.pe_size,
            tanh_exploration=tanh_clipping,
            use_tanh=tanh_clipping > 0,
            n_glimpses=1,
            mask_glimpses=mask_inner,
            mask_logits=mask_logits,
            start_fea = self.start_fea
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def positional_encoding(self, d_model):
        pe_table = []
        for pos in range(self.max_seq_len):
            pos_en = []
            for ii in range(0, int(d_model), 2):
                pos_en.append(math.sin(pos / 10000 ** (2 * ii / d_model)))
                pos_en.append(math.cos(pos / 10000 ** (2 * ii / d_model)))
            pe_table.append(pos_en)
        return torch.FloatTensor(pe_table)

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


    def forward(self, V, V_reach_mask, start_fea, cou_fea, aoi_feature_steps, aoi_start_steps, aoi_pos_steps, aoi_len_steps, aoi_index_steps, E, is_train):

        B = V_reach_mask.size(0)
        N = V_reach_mask.size(1)
        V_dis = V.reshape(B, N, -1)[:, :, [-3, -4]]
        cou_speed = cou_fea.reshape(B, -1)[:, -1]
        V_avg_t = V_dis / cou_speed.unsqueeze(1).unsqueeze(1).repeat(1, N, 1)
        V = torch.cat([V.reshape(B, N, -1), V_avg_t.reshape(B, N, -1)], dim=2)

        # aoi order prediction
        aoi_represent = self.input_layer_aoi(aoi_feature_steps.float())
        aoi_order_score, aoi_order_arg = self.order_decoder_aoi(aoi_represent, aoi_start_steps.float(),
                                                                aoi_len_steps.long())
        aoi_pred_pos = torch.argsort(aoi_order_arg.detach(), dim=1)
        if is_train:
            aoi_sort_info = self.pe_tabel[aoi_pos_steps.long()].float()
        else:
            aoi_sort_info = self.pe_tabel[aoi_pred_pos.long()].float()

        b, seq = aoi_index_steps.shape
        if is_train:
            aoi_real_sort_info = self.pe_tabel[aoi_pos_steps.long()].float()
            aoi_index_order = aoi_index_steps.unsqueeze(-1).expand(b, seq, aoi_real_sort_info.shape[-1]).long()
            match_aoi_order = aoi_real_sort_info.gather(dim=1, index=aoi_index_order)
        else:
            aoi_index_order = aoi_index_steps.unsqueeze(-1).expand(b, seq, aoi_sort_info.shape[-1]).long()
            match_aoi_order = aoi_sort_info.gather(dim=1, index=aoi_index_order)
        ###

        node_h = self.sort_x_embedding(V.float())
        edge_h = self.edge_emb(E)

        package_represent, package_edge = self.unpick_encoder(node_h, edge_h)
        package_represent = torch.cat([package_represent, V.float(), match_aoi_order], dim=-1)

        order_decoder_input = self.start_embed(start_fea.float())

        inputs = package_represent.permute(1, 0, 2).contiguous()
        enc_h = package_represent.permute(1, 0, 2).contiguous()
        dec_init_state = (torch.randn([B, self.hidden_size + self.sort_x_size + self.pe_size], device=V.device).float(), torch.randn([B, self.hidden_size + self.sort_x_size+ self.pe_size], device=V.device).float())
        (pointer_log_scores, order_arg, eta_prediction) = self.decoder(start_fea, order_decoder_input, inputs, dec_init_state, enc_h, V_reach_mask.reshape(-1, N), V)
        pointer_scores = pointer_log_scores.exp()

        return pointer_scores, order_arg, eta_prediction, aoi_order_score


    def model_file_name(self):
        file_name = '+'.join([f'{k}-{self.args[k]}' for k in ['hidden_size']])
        file_name = f'{file_name}.m2g{time.time()}.csv'
        return file_name

# -------------------------------------------------------------------------------------------------------------------------#

from utils.util import save2file_meta
def save2file(params):
    from utils.util import ws
    file_name = ws + f'/output/{params["model"]}.csv'
    head = [
        # data setting
        'dataset', 'min_task_num', 'max_task_num', 'eval_min', 'eval_max',
        # mdoel parameters
        'model', 'hidden_size',
        # training set
        'num_epoch', 'batch_size', 'lr', 'wd', 'early_stop', 'is_test', 'log_time',
        # time metric result
        'mae', 'rmse', 'mape', 'acc_eta@20', 'acc_eta@10', 'acc_eta@30', 'acc_eta@40', 'acc_eta@50', 'acc_eta@60',
        # route metric result
        'lsd', 'lmd', 'krc', 'hr@1', 'hr@2', 'hr@3', 'hr@4', 'hr@5', 'hr@6', 'hr@7', 'hr@8', 'hr@9', 'hr@10',
        'ed', 'acc@1', 'acc@2', 'acc@3', 'acc@4', 'acc@5', 'acc@6', 'acc@7', 'acc@8', 'acc@9', 'acc@10',
    ]
    save2file_meta(params,file_name,head)

