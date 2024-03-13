import torch
import torch.nn as nn
import torch.nn.functional as F


class GAT_layer_multihead(nn.Module):
    def __init__(self,
                 node_in_dim,
                 edge_in_dim,
                 node_out_dim,
                 edge_out_dim,
                 nheads=8,
                 drop=0.5,
                 leaky=0.2,
                 is_mix_attention=True,
                 is_update_edge=True):
        super(GAT_layer_multihead, self).__init__()

        self.nheads = nheads

        self.node_out_dim = node_out_dim
        self.edge_out_dim = edge_out_dim

        self.is_mix_attention = is_mix_attention
        self.is_update_edge = is_update_edge
        if is_mix_attention:
            self.a_edge_init = nn.Linear(edge_in_dim * 1, 1 * nheads)
            self.a_edge = nn.Linear(edge_in_dim * nheads, 1 * nheads)
            if is_update_edge:
                assert edge_out_dim == node_out_dim, "unmatched dimension"
                self.W_edge = nn.Linear(edge_in_dim * nheads, edge_out_dim * nheads)
                self.W_od = nn.Linear(node_in_dim * nheads, node_out_dim * nheads)
                self.W_edge_init = nn.Linear(edge_in_dim * 1, edge_out_dim * nheads)
                self.W_od_init = nn.Linear(node_in_dim * 1, node_out_dim * nheads)
        self.W_init = nn.Linear(node_in_dim * 1, node_out_dim * nheads)
        self.W = nn.Linear(node_in_dim * nheads, node_out_dim * nheads)
        self.a1 = nn.Linear(node_out_dim * nheads, 1 * nheads, bias=False)
        self.a2 = nn.Linear(node_out_dim * nheads, 1 * nheads, bias=False)
        self.leakyrelu = nn.LeakyReLU(leaky)
        self.dropout = nn.Dropout(p=drop)

    def update_edge(self, node_fea, edge_fea, first_layer=False):
        if not first_layer:
            edge = self.W_edge(edge_fea)
            od = self.W_od(node_fea)
        else:
            edge = self.W_edge_init(edge_fea)
            od = self.W_od_init(node_fea)

        o = od.unsqueeze(2)
        d = od.unsqueeze(1)
        return o + d + edge

    @staticmethod
    def mask(x, adj):
        if adj is None:
            return x
        adj = adj.unsqueeze(-1).expand(adj.shape[0], adj.shape[1], adj.shape[2], x.shape[-1])
        x = torch.where(adj > 0, x, -9e15 * torch.ones_like(x))
        return x

    def forward(self, node_fea, edge_fea, adj):
        first_layer = False
        B, H, N, _ = node_fea.shape
        # node_fea: [B, nhead, N, node_in_dim]
        # edge_fea: [B, nhead, N, N, edge_in_dim]

        # node_fea_new: [B, N, nheads * node_in_dim]
        node_fea_new = node_fea.permute(0, 2, 1, 3).reshape(B, N, -1)
        # edge_fea_new: [B, N, N, nheads * node_in_dim]
        edge_fea_new = edge_fea.permute(0, 2, 3, 1, 4).reshape(B, N, N, -1)
        if H == self.nheads:
            # Wh: [B, N, nheads * node_out_dim]
            Wh = self.W(node_fea_new)
            # e_edge: [B, N, N, nheads * 1]
            e_edge = self.a_edge(edge_fea_new)
        else:
            first_layer = True
            # Wh: [B, N, nheads * node_out_dim]
            Wh = self.W_init(node_fea_new)
            # e_edge: [B, N, N, nheads * 1]
            # try:
            e_edge = self.a_edge_init(edge_fea_new)
            # except:
            #     print(self.a_edge_init.weight.shape)
            #     print(edge_fea_new.shape)

        # Whi&j: [B, N, nheads * 1]
        Whi = self.a1(Wh)
        Whj = self.a2(Wh)

        if self.is_mix_attention:
            # e: [B, N, N, nheads * 1]
            e = Whi.unsqueeze(2) + Whj.unsqueeze(1)
            e = self.leakyrelu(e + e_edge)
            e = self.mask(e, adj)
            if self.is_update_edge:
                # edge_fea_new: [B, N, N, nhead * edge_out_dim]
                edge_fea_new = self.update_edge(node_fea_new, edge_fea_new, first_layer)
                # edge_fea_new: [B, N, N, nhead * edge_out_dim]
                #               -> [B, N, N, nhead, edge_out_dim]
                #               -> [B, nhead, N, N, edge_out_dim]
                edge_fea_new = edge_fea_new.reshape(B, N, N, self.nheads, -1).permute(0, 3, 1, 2, 4)
            else:
                edge_fea_new = edge_fea
        else:
            # e: [B, N, N, nheads * 1]
            e = Whi.unsqueeze(2) + Whj.unsqueeze(1)
            e = self.leakyrelu(e)
            e = self.mask(e, adj)
            edge_fea_new = edge_fea

        # attention: [B, nheads, N, N]
        attention = e.permute(0, 3, 1, 2)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout(attention)

        # Wh: [B, nheads, N, node_out_dim]
        Wh = Wh.contiguous().view(B, N, self.nheads, self.node_out_dim).permute(0, 2, 1, 3)
        node_fea_new = torch.matmul(attention, Wh) + Wh

        # node_fea_new: [B, N, node_out_dim]
        # node_fea_new = node_fea_new.mean(dim=1)

        return node_fea_new, edge_fea_new


class GAT_layer(nn.Module):
    def __init__(self, node_size, edge_size, hidden_size, nhead=6, is_mix_attention=False, is_update_edge=True):
        super(GAT_layer, self).__init__()
        self.is_mix_attention = is_mix_attention
        self.is_update_edge = is_update_edge
        if is_mix_attention:
            self.We = nn.Linear(edge_size, hidden_size)
            if is_update_edge:
                self.W_od = nn.Linear(node_size, hidden_size)
        self.W = nn.Linear(node_size, hidden_size)
        self.a1 = nn.Linear(node_size, 1, bias=False)
        self.a2 = nn.Linear(node_size, 1, bias=False)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p=0.5)

    def update_edge(self, node_fea, edge_fea):
        od = self.W_od(node_fea)
        o = od.unsqueeze(2)
        d = od.unsqueeze(1)
        return edge_fea + o + d

    def forward(self, node_fea, edge_fea, adj):
        # node_fea: [B, N, node_size]
        Wh = self.W(node_fea)

        # Whi&j: [B, N, 1]
        Whi = self.a1(Wh)
        Whj = self.a2(Wh)
        # e: [B, N, N]
        if self.is_mix_attention:
            e_node = Whi + Whj.T
            Wh_edge = self.We(edge_fea)
            e = self.leakyrelu(e_node + Wh_edge)
            if adj is not None:
                e = torch.where(adj > 0, e, -9e15 * torch.ones_like(e))
            if self.is_update_edge:
                edge_fea_new = self.update_edge(node_fea, Wh_edge)
            else:
                edge_fea_new = edge_fea
        else:
            e = self.leakyrelu(Whi + Whj.T)
            e = torch.where(adj > 0, e, -9e15 * torch.ones_like(e))
            edge_fea_new = edge_fea

        # attention: [B, N, N]
        attention = F.softmax(e)
        attention = self.dropout(attention)

        node_fea_new = torch.matmul(attention, Wh) + Wh

        return node_fea_new, edge_fea_new


class GAT_encoder(nn.Module):
    def __init__(self, node_size, edge_size, hidden_size,
                 num_layers=3, nheads=4, is_mix_attention=True, is_update_edge=True, num_node=20):
        super(GAT_encoder, self).__init__()
        self.num_layers = num_layers
        self.gat = nn.ModuleList()
        self.gat.append(GAT_layer_multihead(node_in_dim=node_size,
                                            edge_in_dim=edge_size,
                                            node_out_dim=hidden_size,
                                            edge_out_dim=hidden_size,
                                            nheads=nheads,
                                            is_mix_attention=is_mix_attention,
                                            is_update_edge=is_update_edge
                                            ))
        for i in range(1, num_layers):
            self.gat.append(GAT_layer_multihead(node_in_dim=hidden_size,
                                                edge_in_dim=hidden_size,
                                                node_out_dim=hidden_size,
                                                edge_out_dim=hidden_size,
                                                nheads=nheads,
                                                is_mix_attention=is_mix_attention,
                                                is_update_edge=is_update_edge
                                                ))

    def forward(self, node_fea, edge_fea, adj=None):
        node_fea = node_fea.unsqueeze(1)
        edge_fea = edge_fea.unsqueeze(1)
        for i in range(self.num_layers):
            node_fea, edge_fea = self.gat[i](node_fea, edge_fea, adj)
            if i == self.num_layers - 1:

                node_fea = node_fea.mean(dim=1)
            else:
                node_fea = F.relu(node_fea)
        return node_fea, edge_fea
