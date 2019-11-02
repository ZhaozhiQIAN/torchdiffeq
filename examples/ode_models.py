import numpy as np

import torch
import torch.nn as nn


class ODEFunc(nn.Module):

    def __init__(self, eps=100):
        super(ODEFunc, self).__init__()

        self.eps = eps

        self.net = nn.Sequential(
            nn.Linear(4, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 2),
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

