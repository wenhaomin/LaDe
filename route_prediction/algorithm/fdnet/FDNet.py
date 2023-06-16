# -*- coding: utf-8 -*-
import math, time
import numpy as np
from torch.autograd import Variable

from algorithm.fdnet.TP_layers import *

class Decoder(nn.Module):
    def __init__(self,
                 embedding_dim,
                 hidden_dim,
                 eoncode_rnn,
                 args,
                 n_glimpses=1,
                 mask_glimpses=True,
                 mask_logits=True,
                 ):
        super(Decoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_glimpses = n_glimpses
        self.mask_glimpses = mask_glimpses
        self.mask_logits = mask_logits
        self.decode_type = 'greedy' 
        self.rnn = eoncode_rnn
        self.device = args['device']

        self.pointer = Attention(hidden_dim, use_tanh=10 > 0, C=10)
        self.glimpse = Attention(hidden_dim, use_tanh=False)
        self.sm = nn.Softmax(dim=1)
        self.beam_size = 2
        self.mask = None
        self.num_nodes = None

        #beam search
        self.scores = 0
        self.all_scores = []
        # Backpointers at each time-step
        self.prev_Ks = []
        # Outputs at each time-step
        self.next_nodes = []

    def check_mask(self, mask_):
        def mask_modify(mask):
            all_true = mask.all(1)
            mask_mask = torch.zeros_like(mask)
            mask_mask[:, -1] = all_true
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

    def recurrence_bs(self, x, h_in, mask, context):

        logits, h_out = self.calc_logits(x, h_in, mask, context, self.mask_glimpses, self.mask_logits)

        log_p = torch.log_softmax(logits, dim=1) 

        return h_out, log_p

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

        hy, cy = self.rnn(x, h_in)
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

    def tp_update_features(self, V, last_step_index, idxs_select, V_pt,  batch_size, last_step_time):
        V_last = torch.gather(V.permute(1, 0, 2).contiguous(), 0, last_step_index.view(1, batch_size, 1)
                                     .expand(1, batch_size, V.size()[2])).squeeze(0)
        V_current = torch.gather(V.permute(1, 0, 2).contiguous(), 0, idxs_select.view(1, batch_size, 1)
                                        .expand(1, batch_size, V.size()[2])).squeeze(0)

        V_t_left = (V_pt - last_step_time).unsqueeze(2) 
        V_t_left_current = torch.gather(V_t_left.permute(1, 0, 2).contiguous(), 0,
             idxs_select.view(1, batch_size, 1).expand(1, batch_size, V_t_left.size()[2])).squeeze(0)
        tp_input = torch.cat([V_last, V_current, V_t_left_current], dim=1)
        return tp_input

    def rp_update_features(self, idxs_select,  H_update, V_pt,
          b_E, batch_size, current_time, b_E_abs, context):
        decoder_input= torch.gather \
            (context , 0, idxs_select.view(1, batch_size, 1).expand(1, batch_size,  context.size()[2])).squeeze(0)

        b_E_current = torch.gather\
            (b_E.permute(1, 0, 2).contiguous(), 0, idxs_select.view(1, batch_size, 1)
             .expand(1, batch_size, b_E.size()[2])).squeeze(0).unsqueeze(2)
        b_E_abs_current = torch.gather\
            (b_E_abs.permute(1, 0, 2).contiguous(), 0, idxs_select.view(1, batch_size, 1)
             .expand(1, batch_size, b_E.size()[2])).squeeze(0).unsqueeze(2)

        V_t_left = V_pt - current_time

        context_update = torch.cat([H_update.permute(1, 0, 2).contiguous(),  V_t_left.unsqueeze(2),
                             b_E_current, b_E_abs_current], axis = 2).permute(1, 0, 2).contiguous()
        return decoder_input.float(), context_update.float()

    def forward(self, decoder_input, hidden, V_reach_mask, start_idx,
         V_pt, V_ft, b_E, net_TP,
         b_E_abs, V, context_update, H_update, start_fea, context, mode):
  
        if mode == 'beam_search':
            batch_size = V_reach_mask.size()[0]
            self.scores = torch.zeros(batch_size, self.beam_size).to(self.device)
            self.all_scores = []
            self.prev_Ks = []
            self.next_nodes = []
            self.num_nodes = V_reach_mask.size(1)

            time_duration_pred_1 = []
            time_duration_pred_2 = []
            steps = range(self.num_nodes)

            mask_1_ = Variable(V_reach_mask, requires_grad=False)
            mask_2_ = Variable(V_reach_mask, requires_grad=False)
            mask_1 = self.check_mask(mask_1_)
            mask_2 = self.check_mask(mask_2_)

            hidden_1 = hidden
            hidden_2 = hidden

            decoder_input_1 = decoder_input
            decoder_input_2 = decoder_input

            self.mask = torch.stack([mask_1, mask_2], dim=1)

            context_1 = context_update
            context_2 = context_update

            time_prediction = net_TP

            last_step_index_1 = torch.zeros(batch_size).long().to(self.device)
            last_step_index_2 = torch.zeros(batch_size).long().to(self.device)

            current_time_1 = torch.gather \
                (V_ft.unsqueeze(2).permute(1, 0, 2).contiguous(), 0,
                 last_step_index_1.view(1, batch_size, 1).expand(1, batch_size, 1)).squeeze(0) 
            current_time_2 = torch.gather \
                (V_ft.unsqueeze(2).permute(1, 0, 2).contiguous(), 0,
                 last_step_index_2.view(1, batch_size, 1).expand(1, batch_size, 1)).squeeze(0)

            for i in steps:

                hidden_1, log_p_1 = self.recurrence_bs(decoder_input_1, hidden_1, self.mask[:, 0, :], context_1)
                hidden_2, log_p_2 = self.recurrence_bs(decoder_input_2, hidden_2, self.mask[:, 1, :], context_2)
                trans_probs = torch.stack([log_p_1, log_p_2], dim=1)

                beam_lk = trans_probs + self.scores.unsqueeze(2).expand_as(trans_probs)
                if len(self.prev_Ks) == 0:
                    beam_lk[:, 1:] = -1e30 * torch.ones(beam_lk[:, 1:].size())

                beam_lk = beam_lk.view(batch_size, -1)
                bestScores, bestScoresId = beam_lk.topk(self.beam_size, 1, True, True)
                self.scores = bestScores
                prev_k = bestScoresId // self.num_nodes

                self.prev_Ks.append(prev_k)
                new_nodes = bestScoresId - prev_k * self.num_nodes

                self.next_nodes.append(new_nodes)
                perm_mask = prev_k.unsqueeze(2).expand_as(self.mask) 
                self.mask = self.mask.gather(1, perm_mask.long()) 

                idxs_select_1 = new_nodes[:, 0].clone()
                idxs_select_2 = new_nodes[:, 1].clone()

                prev_mask_1 = self.mask[:, 0, :].clone()
                prev_mask_2 = self.mask[:, 1, :].clone()

                self.mask[:, 0, :] = self.update_mask(prev_mask_1, idxs_select_1)
                self.mask[:, 1, :] = self.update_mask(prev_mask_2, idxs_select_2)

                tp_input_1 = self.tp_update_features(V, last_step_index_1, idxs_select_1,
                                                     V_pt, batch_size, current_time_1)

                tp_input_2 = self.tp_update_features(V, last_step_index_2, idxs_select_2,
                                                     V_pt, batch_size, current_time_2)

                pred_time_duration_1 = time_prediction(tp_input_1)
                pred_time_duration_2 = time_prediction(tp_input_2)

                current_time_1 = current_time_1 + pred_time_duration_1
                current_time_2 = current_time_2 + pred_time_duration_2

                decoder_input_1, context_1 = self.rp_update_features \
                    (idxs_select_1, H_update, V_pt,
                     b_E, batch_size, current_time_1,
                     b_E_abs, context)

                decoder_input_2, context_2 = self.rp_update_features \
                    (idxs_select_2, H_update, V_pt,
                     b_E, batch_size, current_time_2,
                     b_E_abs, context)

                last_step_index_1 = idxs_select_1.clone()
                last_step_index_2 = idxs_select_2.clone()

                time_duration_pred_1.append(pred_time_duration_1)
                time_duration_pred_2.append(pred_time_duration_2)

            k = torch.zeros(batch_size, 1).long().to(self.device)
            hyp = -1 * torch.ones(batch_size, self.num_nodes).type(torch.LongTensor)

            for m in range(len(self.prev_Ks) - 1, -2, -1):
                hyp[:, m] = self.next_nodes[m].gather(1, k.long()).view(1, batch_size)
                k = self.prev_Ks[m].gather(1, k.long())

            result_route = hyp

            return result_route, torch.stack(time_duration_pred_1, 1), torch.stack(time_duration_pred_2, 1)

        else:
            batch_size = V_reach_mask.size()[0]
            output_prob = []
            selections = []
            time_duration_pred = []
            steps = range(V_reach_mask.size(1))  
            idxs_select = None
            mask = Variable(V_reach_mask, requires_grad=False) 

            time_prediction = net_TP
            last_step_index = start_idx.long().clone()
   
            current_time = start_fea[:, :, -1].reshape(context_update.size()[1], 1)
            for i in steps:
                hidden, log_p, probs, mask = self.recurrence(decoder_input, hidden, mask, idxs_select, i, context_update)

                _, idxs = probs.max(1)

                if mode == 'teacher_force':
                    idxs_select = idxs

                elif mode == 'greedy':
                    idxs_select = idxs
                tp_input = self.tp_update_features(V, last_step_index, idxs_select, V_pt, batch_size, current_time)
                pred_time_duration = time_prediction(tp_input)

                if mode == 'teacher_force':
                    current_time = current_time + pred_time_duration
                elif mode == 'greedy':
                    current_time = current_time + pred_time_duration

                decoder_input, context_update = self.rp_update_features \
                    ( idxs_select,  H_update, V_pt,
                     b_E, batch_size, current_time, b_E_abs, context_update )

                last_step_index = idxs_select.clone()

                time_duration_pred.append(pred_time_duration.reshape(-1))
                output_prob.append(log_p)
                selections.append(idxs)

            return (torch.stack(output_prob, 1), torch.stack(selections, 1), torch.stack(time_duration_pred, 1))


class Attention(nn.Module):
    def __init__(self, dim, use_tanh=False, C=10):
        super(Attention, self).__init__()
        self.use_tanh = use_tanh
        self.project_query = nn.Linear(dim, dim)
        self.project_ref = nn.Conv1d(dim, dim, 1, 1)
        self.C = C 
        self.tanh = nn.Tanh()
        self.v = nn.Parameter(torch.FloatTensor(dim))
        self.v.data.uniform_(-(1. / math.sqrt(dim)), 1. / math.sqrt(dim))

    def forward(self, query, ref):
        ref = ref.permute(1, 2, 0) 
        q = self.project_query(query).unsqueeze(2)  
        e = self.project_ref(ref)  
        expanded_q = q.repeat(1, 1, e.size(2))
        v_view = self.v.unsqueeze(0).expand(
            expanded_q.size(0), len(self.v)).unsqueeze(1) 
        u = torch.bmm(v_view, self.tanh(expanded_q + e)).squeeze(1) 
        if self.use_tanh:
            logits = self.C * self.tanh(u)
        else:
            logits = u
        return e, logits 


class TimePrediction(nn.Module):
    def __init__(self, args = {}):
        super(TimePrediction, self).__init__()
        self.d_h = args['hidden_size']
        wide_embed_dim = 20
        wide_mlp_dims = (self.d_h, )

        deep_mlp_input_dim = 17
        deep_mlp_dims = (self.d_h, )
        self.device = args['device']
        self.wide_dim = 17
        wide_field_dims = np.array([1] * self.wide_dim)

        self.wide_embedding = FeaturesEmbedding(sum(wide_field_dims), wide_embed_dim)
        self.fm = torch.nn.Sequential(
            FactorizationMachine(),
            torch.nn.BatchNorm1d(wide_embed_dim),
        )
        self.wide_mlp = MultiLayerPerceptron(wide_embed_dim, wide_mlp_dims)
        self.deep_mlp = MultiLayerPerceptron(deep_mlp_input_dim, deep_mlp_dims)
        self.regressor = nn.Linear(self.d_h, 1)


    def forward(self, input_features):
        wide_index = (torch.zeros([input_features.size()[0], self.wide_dim]) + torch.arange(0, self.wide_dim)).long().to(self.device)
        wide_value = input_features
        deep_input = input_features
        wide_embedding = self.wide_embedding(wide_index)
        cross_term = self.fm(wide_embedding.float() * wide_value.unsqueeze(2).float())
        wide_output = self.wide_mlp(cross_term)
        deep_output = self.deep_mlp(deep_input)
        return self.regressor(wide_output + deep_output)


class FDNet(nn.Module):
    def __init__(self, args={}):
        super(FDNet, self).__init__()

        self.args = args
        self.d_h = args['hidden_size']
        self.sort_x_size = 8
        self.d_time = 2
        self.d_spatial = 1
        self.device = args['device']
        self.d_update = 3 
        self.start_dim = 5
        self.start_embed = nn.Linear(self.start_dim, self.d_h + self.d_update)

        self.seq_embedding = nn.Linear(in_features=self.sort_x_size, out_features=self.d_h, bias=False)
        self.sort_x_embedding = nn.Linear(in_features=self.sort_x_size, out_features=self.d_h, bias=False)

        self.start_input_emb = nn.Linear(in_features=self.sort_x_size + 1, out_features=
        self.d_h + self.d_time + self.d_spatial, bias=False)

        self.eoncode_rnn = nn.LSTMCell(self.d_h + self.d_time + self.d_spatial,
                               self.d_h + self.d_time + self.d_spatial)
        self.sort_encoder = MultiLayerPerceptron(input_dim = self.d_h, embed_dims = (self.d_h,))

        self.decoder = Decoder(
            self.d_h + self.d_time + self.d_spatial, 
            self.d_h + self.d_time + self.d_spatial, 
            self.eoncode_rnn,
            args,
            n_glimpses=1,
            mask_glimpses=True,
            mask_logits=True,
        )

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def get_tp_mask(self, max_seq_len, batch_size, sort_len):
        range_tensor = torch.arange(max_seq_len, device=sort_len.device, dtype=sort_len.dtype).expand(batch_size,max_seq_len) 
        each_len_tensor = sort_len.view(-1, 1).expand(batch_size, max_seq_len) 
        raw_mask_tensor = range_tensor < each_len_tensor
        return raw_mask_tensor + 0

    def enc_sort_emb(self, sort_emb):
        sort_encoder_outputs = self.sort_encoder(sort_emb)
        H_update = sort_encoder_outputs.permute(1, 0, 2).contiguous()  
        H = sort_encoder_outputs.permute(1, 0, 2).contiguous()  

        return H_update, H

    def get_init_features(self, B, N, T, V, V_pt, E, V_ft, start_idx, V_reach_mask, E_abs, V_dispatch_mask, E_mask, start_fea):

        b_E = torch.zeros([B, T, N, N]).to(self.device)
        b_E_abs = torch.zeros([B, T, N, N]).to(self.device)
        b_V_update = torch.zeros([B, T, N, self.d_update]).to(self.device)
        decoder_input = torch.zeros([B, T, self.d_h + self.d_update]).to(self.device)

        for t in range(T):
            E_update = torch.gather(E.permute(1, 0, 2).contiguous(), 0, start_idx[:, t].view(1, B, 1).
                                  expand(1, B, E.size()[2])).squeeze(0)  
            E_abs_update = torch.gather(E_abs.permute(1, 0, 2).contiguous(), 0, start_idx[:, t].view(1, B, 1).
                                  expand(1, B, E_abs.size()[2])).squeeze(0)  
            E_update = E_update * V_dispatch_mask[:, t, :]
            E_abs_update = E_abs_update * V_dispatch_mask[:, t, :]
            current_time = start_fea[:, t, -1].unsqueeze(1)
            E_mask_t = E_mask[:, t, :, :].float()
            decoder_input[:, t, :] = self.start_embed(start_fea[:, t].float())
            V_update = torch.cat([E_update.unsqueeze(2), E_abs_update.unsqueeze(2), (V_pt[:, t, :] - current_time).unsqueeze(2)], dim=2)
            b_E[:, t, :, :] =  E * E_mask_t
            b_E_abs[:, t, :, :] = E_abs * E_mask_t
            b_V_update[:, t, :, :] = V_update
        b_init_hx = torch.randn(B * T, self.d_h + self.d_update).to(self.device)
        b_init_cx = torch.randn(B * T, self.d_h + self.d_update).to(self.device)

        return V.reshape(B*T, N, -1), V_reach_mask.reshape(-1, N), b_E.reshape(B * T, N, N), \
               b_E_abs.reshape(B * T, N, N), b_V_update, (b_init_hx, b_init_cx),\
               V_ft.reshape(B * T, N, 1), V_pt.reshape(B*T, N, -1), \
               decoder_input.reshape(B * T, self.d_h + self.d_update)

    def forward(self, V, E, V_pt, V_ft, target_len, V_reach_mask, target, net_TP,
                        start_idx, E_abs, E_mask, V_dispatch_mask, start_fea, mode):

        B, T, N = V_reach_mask.shape
        target = target.reshape(B * T, N)
        V, V_reach_mask, b_E, b_E_abs, b_V_update, hidden, V_ft, V_pt,decoder_input = self.get_init_features(B, N, T, V,
                                                V_pt, E, V_ft, start_idx.long(), V_reach_mask, E_abs,
                                                V_dispatch_mask, E_mask, start_fea)

        sort_x_emb = self.sort_x_embedding(V.float())

        H_update, H = self.enc_sort_emb(sort_x_emb)

        context_update = torch.cat([H_update.reshape(B * T, N, self.d_h), b_V_update.reshape(B * T, N, self.d_update)], dim=2). \
            permute(1, 0, 2).contiguous().clone()

        context = torch.cat([H.reshape(B * T, N, self.d_h), b_V_update.reshape(B * T, N, self.d_update)], dim=2). \
            permute(1, 0, 2).contiguous().clone()

        t_mask = self.get_tp_mask(N, B * T, target_len.reshape(B * T))
        if mode == 'beam_search':
            (result_route, td_pred_1, td_pred_2) = \
                self.decoder(decoder_input, hidden, V_reach_mask, start_idx.reshape(-1).float(),
                             V_pt.squeeze(2), V_ft.squeeze(2), b_E, net_TP,
                         b_E_abs, V, context_update, H_update.reshape(N, B*T, self.d_h), start_fea, context , mode)

            return result_route

        elif (mode == 'teacher_force'):

            (result_route_scores, result_route, td_pred) = \
            self.decoder(decoder_input, hidden, V_reach_mask, start_idx.reshape(-1).float(),
                             V_pt.squeeze(2), V_ft.squeeze(2), b_E, net_TP,
                         b_E_abs, V, context_update, H_update.reshape(N, B*T, self.d_h), start_fea, context, mode)

            return result_route_scores.exp(), result_route, td_pred * t_mask

        elif (mode == 'greedy'):
            (result_route_scores, result_route, td_pred) = \
                self.decoder(decoder_input, hidden, V_reach_mask, start_idx.reshape(-1).float(),
                             V_pt.squeeze(2), V_ft.squeeze(2), b_E, net_TP,
                         b_E_abs, V, context_update, H_update.reshape(N, B*T, self.d_h), start_fea, context, mode)

            return result_route
        else:
            raise Exception('please assign a prediction method: greedy or beam_search')

    def model_file_name(self):
        file_name = '+'.join([f'{k}-{self.args[k]}' for k in ['hidden_size']])
        file_name = f'{file_name}.route_prediction_fdnet_{time.time()}'
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
        'num_epoch', 'batch_size', 'lr', 'wd', 'early_stop',  'is_test', 'log_time',
        # metric result
        'lsd', 'lmd', 'krc',  'hr@1', 'hr@2', 'hr@3', 'hr@4', 'hr@5', 'hr@6', 'hr@7', 'hr@8', 'hr@9', 'hr@10',
        'ed','acc@1', 'acc@2', 'acc@3', 'acc@4', 'acc@5', 'acc@6', 'acc@7', 'acc@8', 'acc@9', 'acc@10',
    ]
    save2file_meta(params,file_name,head)
