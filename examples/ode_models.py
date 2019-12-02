import numpy as np

import torch
import torch.nn as nn


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


class ODEFunc0(nn.Module):

    def __init__(self, dim_y=2):
        super(ODEFunc0, self).__init__()

        self.dim_y = dim_y

        self.net = nn.Sequential(
            nn.Linear(self.dim_y, 50),
            nn.Tanh(),
            nn.Linear(50, self.dim_y),
        )

        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y)


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
