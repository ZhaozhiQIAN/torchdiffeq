import numpy as np

import torch
import torch.nn as nn


class ODEFunc(nn.Module):

    def __init__(self):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(4, 50),
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
        return self.net(y_expand) * torch.tensor([1./ (eps / 10), 1.])
