import os
import torch
import numpy as np
import torch.utils.data
from src.utils.scaler import StandardScaler
from src.utils import controldiffeq
import torch.nn as nn
import random
import torch.nn.functional as F

class NeuralGCDE(nn.Module):
    def __init__(self, args, func_f, func_g, input_channels, hidden_channels, output_channels, initial, device, atol, rtol, solver):
        super(NeuralGCDE, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = input_channels
        self.hidden_dim = hidden_channels
        self.output_dim = output_channels
        self.horizon = args.horizon
        self.num_layers = args.num_layers

        self.default_graph = args.default_graph
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)
        
        self.func_f = func_f
        self.func_g = func_g
        self.solver = solver
        self.atol = atol
        self.rtol = rtol

        #predictor
        self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

        self.init_type = 'fc'
        if self.init_type == 'fc':
            self.initial_h = torch.nn.Linear(self.input_dim, self.hidden_dim)
            self.initial_z = torch.nn.Linear(self.input_dim, self.hidden_dim)
        elif self.init_type == 'conv':
            self.start_conv_h = nn.Conv2d(in_channels=input_channels,
                                            out_channels=hidden_channels,
                                            kernel_size=(1,1))
            self.start_conv_z = nn.Conv2d(in_channels=input_channels,
                                            out_channels=hidden_channels,
                                            kernel_size=(1,1))
        self.drop = nn.Dropout(0.1)
        print('Use Dropout 0.1')

    def forward(self, times, coeffs):
        #source: B, T_1, N, D
        #target: B, T_2, N, D
        spline = controldiffeq.NaturalCubicSpline(times, coeffs)
        if self.init_type == 'fc':
            h0 = self.initial_h(spline.evaluate(times[0]))
            z0 = self.initial_z(spline.evaluate(times[0]))
        elif self.init_type == 'conv':
            h0 = self.start_conv_h(spline.evaluate(times[0]).transpose(1,2).unsqueeze(-1)).transpose(1,2).squeeze()
            z0 = self.start_conv_z(spline.evaluate(times[0]).transpose(1,2).unsqueeze(-1)).transpose(1,2).squeeze()

        z_t = controldiffeq.cdeint_gde_dev(dX_dt=spline.derivative, #dh_dt
                                   h0=h0,
                                   z0=z0,
                                   func_f=self.func_f,
                                   func_g=self.func_g,
                                   t=times,
                                   method=self.solver,
                                   atol=self.atol,
                                   rtol=self.rtol)
        z_T = z_t[-1:,...].transpose(0,1)
        
        z_T = self.drop(z_T)

        #CNN based predictor
        output = self.end_conv(z_T)                         #B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)                             #B, T, N, C

        return output

class FinalTanh_f(nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers):
        super(FinalTanh_f, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.linear_in = nn.Linear(hidden_channels, hidden_hidden_channels)
        
        self.linears = nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
                                           for _ in range(num_hidden_layers - 1))
        self.linear_out = nn.Linear(hidden_hidden_channels, input_channels * hidden_channels) #32,32*4  -> # 32,32,4 

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z):
        z = self.linear_in(z)
        z = z.relu()

        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.input_channels)    
        z = z.tanh()
        return z

class FinalTanh_f_prime(nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers):
        super(FinalTanh_f_prime, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.linear_in = nn.Linear(hidden_channels, hidden_hidden_channels)
        
        self.linears = nn.ModuleList(torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
                                           for _ in range(num_hidden_layers - 1))
        self.linear_out = nn.Linear(hidden_hidden_channels, hidden_channels * hidden_channels) #32,32*4  -> # 32,32,4 

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z):
        z = self.linear_in(z)
        z = z.relu()

        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.hidden_channels)    
        z = z.tanh()
        return z

class FinalTanh_f2(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers):
        super(FinalTanh_f2, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.start_conv = torch.nn.Conv2d(in_channels=hidden_channels,
                                    out_channels=hidden_channels,
                                    kernel_size=(1,1))

        self.linears = torch.nn.ModuleList(torch.nn.Conv2d(in_channels=hidden_channels,
                                    out_channels=hidden_channels,
                                    kernel_size=(1,1))
                                           for _ in range(num_hidden_layers - 1))
        
        self.linear_out = torch.nn.Conv2d(in_channels=hidden_channels,
                                    out_channels=input_channels*hidden_channels,
                                    kernel_size=(1,1))

    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z):
        z = self.start_conv(z.transpose(1,2).unsqueeze(-1))
        z = z.relu()

        for linear in self.linears:
            z = linear(z)
            z = z.relu()

        z = self.linear_out(z).squeeze().transpose(1,2).view(*z.transpose(1,2).shape[:-2], self.hidden_channels, self.input_channels)
        z = z.tanh()
        return z

class VectorField_g(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers, num_nodes, cheb_k, embed_dim,
                    g_type):
        super(VectorField_g, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels)
        
        self.linear_out = torch.nn.Linear(hidden_hidden_channels, hidden_channels * hidden_channels)
        
        self.g_type = g_type
        if self.g_type == 'agc':
            self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
            self.cheb_k = cheb_k
            self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, hidden_hidden_channels, hidden_hidden_channels))
            self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, hidden_hidden_channels))


    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z):
        z = self.linear_in(z)
        z = z.relu()

        if self.g_type == 'agc':
            z = self.agc(z)
        else:
            raise ValueError('Check g_type argument')

        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.hidden_channels)
        z = z.tanh()
        return z 

    def agc(self, z):
        """
        Adaptive Graph Convolution
        - Node Adaptive Parameter Learning
        - Data Adaptive Graph Generation
        """
        node_num = self.node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)
        # laplacian=False
        laplacian=False
        if laplacian == True:
            support_set = [supports, -torch.eye(node_num).to(supports.device)]

        else:
            support_set = [torch.eye(node_num).to(supports.device), supports]
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', self.node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(self.node_embeddings, self.bias_pool)                       #N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, z)      #B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        z = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     #b, N, dim_out
        return z


class VectorField_only_g(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers, num_nodes, cheb_k, embed_dim,
                    g_type):
        super(VectorField_only_g, self).__init__()
        
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels)
        
        self.linear_out = torch.nn.Linear(hidden_hidden_channels, input_channels * hidden_channels) #32,32*4  -> # 32,32,4 

        
        self.g_type = g_type
        if self.g_type == 'agc':
            self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
            self.cheb_k = cheb_k
            self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, hidden_hidden_channels, hidden_hidden_channels))
            self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, hidden_hidden_channels))


    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z):
        z = self.linear_in(z)
        z = z.relu()

        if self.g_type == 'agc':
            z = self.agc(z)
        else:
            raise ValueError('Check g_type argument')

        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.input_channels)
        z = z.tanh()
        return z #torch.Size([64, 307, 64, 1])

    def agc(self, z):
        """
        Adaptive Graph Convolution
        - Node Adaptive Parameter Learning
        - Data Adaptive Graph Generation
        """
        node_num = self.node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)

        laplacian=False
        if laplacian == True:
            support_set = [supports, -torch.eye(node_num).to(supports.device)]
        else:
            support_set = [torch.eye(node_num).to(supports.device), supports]

        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', self.node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(self.node_embeddings, self.bias_pool)                       #N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, z)      #B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        z = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     #b, N, dim_out
        return z

class VectorField_g_prime(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, hidden_hidden_channels, num_hidden_layers, num_nodes, cheb_k, embed_dim,
                    g_type):
        super(VectorField_g_prime, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers
        self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels)
        self.linear_out = torch.nn.Linear(hidden_hidden_channels, input_channels * hidden_channels) #32,32*4  -> # 32,32,4 
        
        self.g_type = g_type
        if self.g_type == 'agc':
            self.node_embeddings = nn.Parameter(torch.randn(num_nodes, embed_dim), requires_grad=True)
            self.cheb_k = cheb_k
            self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, hidden_hidden_channels, hidden_hidden_channels))
            self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, hidden_hidden_channels))


    def extra_repr(self):
        return "input_channels: {}, hidden_channels: {}, hidden_hidden_channels: {}, num_hidden_layers: {}" \
               "".format(self.input_channels, self.hidden_channels, self.hidden_hidden_channels, self.num_hidden_layers)

    def forward(self, z):
        z = self.linear_in(z)
        z = z.relu()

        if self.g_type == 'agc':
            z = self.agc(z)
        else:
            raise ValueError('Check g_type argument')

        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.input_channels)
        z = z.tanh()
        return z #torch.Size([64, 307, 64, 1])

    def agc(self, z):
        """
        Adaptive Graph Convolution
        - Node Adaptive Parameter Learning
        - Data Adaptive Graph Generation
        """
        node_num = self.node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]
        #default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', self.node_embeddings, self.weights_pool)  #N, cheb_k, dim_in, dim_out
        bias = torch.matmul(self.node_embeddings, self.bias_pool)                       #N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, z)      #B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        z = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     #b, N, dim_out
        return z
    


def make_model(args):
    if args.model_type == 'type1':
        vector_field_f = FinalTanh_f(input_channels=args.input_dim, hidden_channels=args.hid_dim,
                                        hidden_hidden_channels=args.hid_hid_dim,
                                        num_hidden_layers=args.num_layers)
        vector_field_g = VectorField_g(input_channels=args.input_dim, hidden_channels=args.hid_dim,
                                        hidden_hidden_channels=args.hid_hid_dim,
                                        num_hidden_layers=args.num_layers, num_nodes=args.num_nodes, cheb_k=args.cheb_k, embed_dim=args.embed_dim,
                                        g_type=args.g_type)
        model = NeuralGCDE(args, func_f=vector_field_f, func_g=vector_field_g, input_channels=args.input_dim, hidden_channels=args.hid_dim,
                                        output_channels=args.output_dim, initial=True,
                                        device=args.device, atol=1e-9, rtol=1e-7, solver=args.solver)
        return model, vector_field_f, vector_field_g

def print_model_parameters(model, only_num = True):
    print('*****************Model Parameter*****************')
    if not only_num:
        for name, param in model.named_parameters():
            print(name, param.shape, param.requires_grad)
    total_num = sum([param.nelement() for param in model.parameters()])
    print('Total params num: {}'.format(total_num))
    print('*****************Finish Parameter****************')
    return total_num

def init_seed(seed):
    '''
    Disable cudnn to maximize reproducibility
    '''
    torch.cuda.cudnn_enabled = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def get_dataloader_cde(args, normalizer = 'std', tod=False, dow=False, weather=False, single=True):
     
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(args.dataset, category + '.npz'))
        data['x_' + category] = cat_data['x'][..., :args.input_dim - 1]
        data['y_' + category] = cat_data['y'][..., :args.output_dim]

    if args.dataset_name[:8] == 'Delivery':
        scaler = []
        for i in range(data['x_train'].shape[2]):
            scaler.append(StandardScaler(mean=data['x_train'][:,:,i, 0].mean(),
                                        std=data['x_train'][:,:,i, 0].std()))
        for category in ['train', 'val', 'test']:
            for i in range(data['x_train'].shape[2]):
                data['x_' + category][:,:,i, :args.input_dim - 1] = scaler[i].transform(data['x_' + category][:,:,i, :args.input_dim - 1])
                data['y_' + category][:,:,i, :args.output_dim] = scaler[i].transform(data['y_' + category][:,:,i, :args.output_dim])
    else:
        scaler = StandardScaler(mean=data['x_train'][..., :args.input_dim - 1].mean(), std=data['x_train'][..., :args.input_dim - 1].std())
        for category in ['train', 'val', 'test']:
            data['x_' + category][..., :args.input_dim - 1] = scaler.transform(data['x_' + category][..., :args.input_dim - 1])
            data['y_' + category][..., :args.output_dim] = scaler.transform(data['y_' + category][..., :args.output_dim])

    x_tra, y_tra = data['x_train'], data['y_train']
    x_val, y_val = data['x_val'], data['y_val']
    x_test, y_test = data['x_test'], data['y_test']
    print('Train: ', x_tra.shape, y_tra.shape)
    print('Val: ', x_val.shape, y_val.shape)
    print('Test: ', x_test.shape, y_test.shape)


    times = torch.linspace(0, int(x_tra.shape[1])-1, x_tra.shape[1] )

    augmented_X_tra = []
    augmented_X_tra.append(times.unsqueeze(0).unsqueeze(0).repeat(x_tra.shape[0],x_tra.shape[2],1).unsqueeze(-1).transpose(1,2))
    augmented_X_tra.append(torch.Tensor(x_tra[..., :]))
    x_tra = torch.cat(augmented_X_tra, dim=3)
    augmented_X_val = []
    augmented_X_val.append(times.unsqueeze(0).unsqueeze(0).repeat(x_val.shape[0],x_val.shape[2],1).unsqueeze(-1).transpose(1,2))
    augmented_X_val.append(torch.Tensor(x_val[..., :]))
    x_val = torch.cat(augmented_X_val, dim=3)
    augmented_X_test = []
    augmented_X_test.append(times.unsqueeze(0).unsqueeze(0).repeat(x_test.shape[0],x_test.shape[2],1).unsqueeze(-1).transpose(1,2))
    augmented_X_test.append(torch.Tensor(x_test[..., :]))
    x_test = torch.cat(augmented_X_test, dim=3)

    train_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, x_tra.transpose(1,2))
    valid_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, x_val.transpose(1,2))
    test_coeffs = controldiffeq.natural_cubic_spline_coeffs(times, x_test.transpose(1,2))

    ##############get dataloader######################
    train_dataloader = data_loader_cde(train_coeffs, y_tra, args.batch_size, shuffle=True, drop_last=True)

    if len(x_val) == 0:
        val_dataloader = None
    else:
        val_dataloader = data_loader_cde(valid_coeffs, y_val, args.batch_size, shuffle=False, drop_last=True)

    test_dataloader = data_loader_cde(test_coeffs, y_test, args.batch_size, shuffle=False, drop_last=False)
    return train_dataloader, val_dataloader, test_dataloader, scaler, times

def data_loader_cde(X, Y, batch_size, shuffle=True, drop_last=True):
    cuda = True if torch.cuda.is_available() else False
    TensorFloat = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    data = torch.utils.data.TensorDataset(*X, torch.tensor(Y))
    dataloader = torch.utils.data.DataLoader(data, batch_size=batch_size,
                                             shuffle=shuffle, drop_last=drop_last)
    return dataloader