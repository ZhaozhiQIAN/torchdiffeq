import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class ODEBase(nn.Module):
    def __init__(self):
        super(ODEBase, self).__init__()
        self.counter_list = list()
        self.counter = 0

    def reset_counter(self):
        self.counter_list.append(self.counter)
        self.counter = 0


class AttentiveODE(ODEBase):
    def __init__(self, dim_y=2, dim_hidden=None):
        super(AttentiveODE, self).__init__()

        self.dim_y = dim_y
        self.dim_hidden = dim_hidden
        self.gru_hidden = None

        if dim_hidden is not None:
            self.net = nn.Sequential(
                nn.Linear(self.dim_y * 2, self.dim_hidden),
                nn.Tanh(),
                nn.Linear(self.dim_hidden, self.dim_y),
            )
        else:
            self.net = nn.Linear(self.dim_y * 2, self.dim_y)

        self.attn = nn.Linear(dim_y * 2, 1)

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def set_gru_hidden(self, gru_hidden):
        self.gru_hidden = gru_hidden

    def forward(self, t, y):
        self.counter += 1
        attn_input = torch.cat((self.gru_hidden, y.repeat((1, self.gru_hidden.shape[1], 1))), 2)
        attn_linear = self.attn(attn_input)
        attn_score = F.softmax(attn_linear, dim=1)
        context_vector = torch.bmm(attn_score.permute((0, 2, 1)), self.gru_hidden)
        ode_input = torch.cat((y, context_vector), dim=2)
        return self.net(ode_input)


class ODEFunc0(ODEBase):

    def __init__(self, dim_y=2, dim_hidden=50):
        super(ODEFunc0, self).__init__()

        self.dim_y = dim_y
        self.dim_hidden = dim_hidden

        self.net = nn.Sequential(
            nn.Linear(self.dim_y, self.dim_hidden),
            nn.Tanh(),
            nn.Linear(self.dim_hidden, self.dim_y),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        self.counter += 1
        return self.net(y)


class ODELinear(ODEBase):

    def __init__(self, dim_y=2):
        super(ODELinear, self).__init__()

        self.dim_y = dim_y
        self.lin = nn.Linear(self.dim_y, self.dim_y)

    def forward(self, t, y):
        self.counter += 1
        return self.lin(y)


class FSODE(ODEBase):
    def __init__(self, input_dim=4, hidden_dim=20):
        super(FSODE, self).__init__()
        # slow net
        self.net_y = LatentBlock(input_dim, hidden_dim)
        self.lin_y = nn.Linear(hidden_dim, input_dim)

        # fast net
        self.net_x1 = LatentBlock(input_dim, hidden_dim)
        self.net_x2 = LatentBlock(hidden_dim, hidden_dim)
        self.lin_x = nn.Linear(hidden_dim, input_dim)

    def forward(self, t, y):
        self.counter += 1

        # slow net
        y_out = self.net_y(y)
        y_out_final = self.lin_y(y_out)

        # fast net
        x1_out = self.net_x1(y)
        x2_in = y_out + x1_out
        x2_out = self.net_x2(x2_in)
        x_out_final = self.lin_x(x2_out)

        return x_out_final + y_out_final

    def forward_slow(self, t, y):
        y_out = self.net_y(y)
        y_out_final = self.lin_y(y_out)
        return y_out_final

    def forward_fast(self, t, y):
        y_out = self.net_y(y)
        x1_out = self.net_x1(y)
        x2_in = y_out + x1_out
        x2_out = self.net_x2(x2_in)

        x_out_final = self.lin_x(x2_out)

        return x_out_final


class HigherOrderOdeLatent(ODEBase):

    def __init__(self, dim=2, hidden_size=50):
        super(HigherOrderOdeLatent, self).__init__()

        assert dim % 2 == 0

        self.order = 2
        self.dim = dim // self.order

        self.net = nn.Sequential(
            nn.Linear(self.dim * self.order, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, self.dim),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        self.counter += 1

        output_list = []

        y_i = y[..., self.dim:]
        output_list.append(y_i)

        output_list.append(self.net(y))

        return torch.cat(output_list, axis=len(output_list[0].shape) - 1)


class EncoderLSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EncoderLSTM, self).__init__()
        self.hidden_dim = hidden_dim

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(input_dim + 1, hidden_dim)

        # The linear layer that maps from hidden state space to output space
        self.lin = nn.Linear(hidden_dim, output_dim)

    def forward(self, y, t):
        # y and t are the first k observations

        batch_size = y.shape[1]
        dim_y = y.shape[-1]
        t_max = t.shape[0]

        t = t.view((t_max, batch_size, 1))
        y = y.squeeze()

        y_in = torch.cat((y, t), dim=-1)

        hidden = None

        for t in reversed(range(t_max)):
            obs = y_in[t:t + 1, ...]
            out, hidden = self.lstm(obs, hidden)
        out_linear = self.lin(out)

        return out_linear


class LatentBlock(nn.Module):

    def __init__(self, input_dim=4, nhidden=20):
        super(LatentBlock, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, nhidden),
            nn.Tanh(),
            nn.Linear(nhidden, nhidden),
        )

    def forward(self, x):
        return self.net(x)


class HigherOrderOde(nn.Module):

    def __init__(self, dat_dict, batch_size=1, dim=2, order=2, hidden_size=50):
        super(HigherOrderOde, self).__init__()

        self.dim = dim
        self.order = order
        self.batch_size = batch_size
        # assert self.batch_size == 1

        eid_list = list(dat_dict.keys())
        self.eid_to_id = dict(zip(eid_list, range(len(eid_list))))

        # imitate the size of batch_y0
        init_mat = np.zeros((len(eid_list), 1, dim * order))
        for eid, v in dat_dict.items():
            t = v['t'].numpy()
            x = v['x'].numpy()
            idx = self.eid_to_id[eid]
            init_mat[idx, 0, :dim] = x[0, ...]

            if order > 1 and len(t) > 1:
                init_mat[idx, 0, dim:(2 * dim + 1)] = (x[1, ...] - x[0, ...]) / (t[1] - t[0])

        self.init_cond_mat = torch.nn.Parameter(torch.tensor(init_mat, dtype=torch.float32))
        self.init_cond = None

        self.net = nn.Sequential(
            nn.Linear(self.dim * self.order, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, self.dim),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def set_init_cond(self, eids):
        if len(eids) == 1:
            idx = self.eid_to_id[eids[0]]
            self.init_cond = self.init_cond_mat[idx:idx+1, ...]
        else:
            idx = np.array([self.eid_to_id[x] for x in eids])
            id_torch = torch.from_numpy(idx)
            self.init_cond = self.init_cond_mat[id_torch, ...]

    def forward(self, t, y):
        y = y + self.init_cond
        output_list = []

        y_i = y[..., self.dim:]
        output_list.append(y_i)

        output_list.append(self.net(y))

        return torch.cat(output_list, axis=len(output_list[0].shape) - 1)


class HigherOrderOdeV2(nn.Module):

    def __init__(self, dat_dict, batch_size=1, dim=2, order=2, hidden_size=50):
        super(HigherOrderOdeV2, self).__init__()

        self.dim = dim
        self.order = order
        self.batch_size = batch_size
        assert self.order == 2

        eid_list = list(dat_dict.keys())
        self.eid_to_id = dict(zip(eid_list, range(len(eid_list))))

        self.init_x = nn.ParameterList()
        self.init_v = nn.ParameterList()

        for eid in eid_list:
            t = dat_dict[eid]['t'].numpy()
            x = dat_dict[eid]['x'].numpy()
            self.init_x.append(torch.nn.Parameter(torch.tensor(x[0:1, ...], dtype=torch.float32)))
            v = (x[1:2, ...] - x[0:1, ...]) / (t[1] - t[0])
            self.init_v.append(torch.nn.Parameter(torch.tensor(v, dtype=torch.float32)))

        self.init_cond = None

        self.net = nn.Sequential(
            nn.Linear(self.dim * self.order, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, self.dim),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def set_init_cond(self, eids):
        cond_list = []
        for eid in eids:
            idx = self.eid_to_id[eid]
            cond_list.append(torch.cat((self.init_x[idx], self.init_v[idx]), dim=2))
        self.init_cond = torch.cat(cond_list, dim=0)

    def forward(self, t, y):
        y = y + self.init_cond
        output_list = []

        y_i = y[..., self.dim:]
        output_list.append(y_i)

        output_list.append(self.net(y))

        return torch.cat(output_list, axis=len(output_list[0].shape) - 1)


class ODEFunc(nn.Module):

    def __init__(self, eps=100, dim_y=2):
        super(ODEFunc, self).__init__()

        self.eps = eps
        self.dim_y = dim_y

        self.net = nn.Sequential(
            nn.Linear(self.dim_y * 2, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, self.dim_y),
        )

        # self.net = nn.Sequential(
        #     nn.Linear(4, 2)
        # )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        # print(y.shape)
        y_expand = torch.cat((y, y**3), axis=-1)
        # print(y_expand.shape)

        out = self.net(y_expand)
        # print(out.shape)
        return out * torch.tensor([1./ (self.eps / 10), 1.])


class ODEFuncAug(nn.Module):

    def __init__(self, dim_y=2, dim_aug=2):
        super(ODEFuncAug, self).__init__()

        self.dim_y = dim_y
        self.dim_aug = dim_aug

        self.net = nn.Sequential(
            nn.Linear(self.dim_y + self.dim_aug, 50),
            nn.Tanh(),
            nn.Linear(50, self.dim_y + self.dim_aug),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)


class ODEFuncLayered(nn.Module):

    def __init__(self, eps=100):
        super(ODEFuncLayered, self).__init__()

        self.eps = eps

        self.net_y = nn.Sequential(
            nn.Linear(4, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
        )

        self.net_x = nn.Sequential(
            nn.Linear(4, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1),
        )

        for m in self.net_x.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
        
        for m in self.net_y.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        y_expand = torch.cat((y, y**3), axis=-1)

        y_out = self.net_y(y_expand)
        x_out = self.net_x(y_expand)

        # print(torch.cat((x_out, y_out), -1).shape)

        return torch.cat((x_out, y_out), -1)


class ODEFuncLayeredCombined(nn.Module):

    def __init__(self, eps=100):
        super(ODEFuncLayeredCombined, self).__init__()

        self.eps = eps

        self.net_y = nn.Sequential(
            nn.Linear(4, 50),
            nn.Tanh(),
            nn.Linear(50, 2),
        )

        self.net_x = nn.Sequential(
            nn.Linear(4, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 2),
        )

        for m in self.net_x.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
        
        for m in self.net_y.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        y_expand = torch.cat((y, y**3), axis=-1)

        y_out = self.net_y(y_expand)
        x_out = self.net_x(y_expand)

        # print(torch.cat((x_out, y_out), -1).shape)

        return x_out + y_out



class ODEFuncLayeredResidual(nn.Module):

    def __init__(self, eps=100):
        super(ODEFuncLayeredResidual, self).__init__()

        self.eps = eps

        self.net_y = nn.Sequential(
            nn.Linear(4, 50),
            nn.Tanh(),
        )

        self.lin_y = nn.Linear(50, 2)

        self.net_x1 = nn.Sequential(
            nn.Linear(4, 50),
            nn.Tanh(),
        )

        self.net_x2 = nn.Sequential(
            nn.Linear(50, 50),
            nn.Tanh(),
        )

        self.lin_x = nn.Linear(50, 2)

        for n in [self.net_y, self.net_x1, self.net_x2]:
            for m in n.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=0.1)
                    nn.init.constant_(m.bias, val=0)
        
    def forward(self, t, y):
        y_expand = torch.cat((y, y**3), axis=-1)

        y_out = self.net_y(y_expand)
        x1_out = self.net_x1(y_expand)

        x2_in = y_out + x1_out
        x2_out = self.net_x2(x2_in)

        y_out_final = self.lin_y(y_out)
        x_out_final = self.lin_x(x2_out)

        # print(torch.cat((x_out, y_out), -1).shape)

        return x_out_final + y_out_final

    def forward_slow(self, t, y):
        y_expand = torch.cat((y, y ** 3), axis=-1)

        y_out = self.net_y(y_expand)
        y_out_final = self.lin_y(y_out)
        return y_out_final

    def forward_fast(self, t, y):
        y_expand = torch.cat((y, y ** 3), axis=-1)

        y_out = self.net_y(y_expand)
        x1_out = self.net_x1(y_expand)

        x2_in = y_out + x1_out
        x2_out = self.net_x2(x2_in)

        x_out_final = self.lin_x(x2_out)

        return x_out_final


class ODEFuncLayeredResidualSlowOnly(ODEFuncLayeredResidual):
    def forward(self, t, y):
        return super(ODEFuncLayeredResidualSlowOnly, self).forward_slow(t, y)


class ODEFuncLayeredResidualFastOnly(ODEFuncLayeredResidual):
    def forward(self, t, y):
        return super(ODEFuncLayeredResidualFastOnly, self).forward_fast(t, y)
