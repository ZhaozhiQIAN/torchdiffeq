import time
import random
import numpy as np
import pandas as pds
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchdiffeq import odeint_adjoint as odeint

from config import D_TYPE


def get_data():
    Arg = namedtuple('Arg', ['method', 'data_size', 'batch_time', 'batch_size',
                             'niters', 'test_freq', 'viz', 'gpu', 'adjoint'])
    args = Arg('dopri5', 1000, 20, 1, 2000, 50, False, 1, True)

    true_y0 = torch.tensor([[2.0, 1.0]], dtype=D_TYPE)
    t = torch.linspace(0, 10, args.data_size, dtype=D_TYPE)

    class Lambda(nn.Module):

        def __init__(self, gamma=1., beta=0.5):
            super(Lambda, self).__init__()
            self.gamma = gamma
            self.beta = beta

        def forward(self, t, y):
            y1 = y[:, 0]
            y2 = y[:, 1]
            dy1_dt = y2
            dy2_dt = - self.beta * y2 - self.gamma * y1

            return torch.cat((dy1_dt.reshape(-1, 1), dy2_dt.reshape(-1, 1)), axis=1)

    # This numerical solution given the true DE.
    with torch.no_grad():
        true_y = odeint(Lambda(), true_y0, t, method='dopri5')
    true_y = true_y[..., 0:1]

    dat_dict = dict()

    for s in range(args.data_size - args.batch_time):
        batch_t = t[:args.batch_time]
        batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)
        dim = batch_y.shape[-1]
        x_reshaped = batch_y.reshape(args.batch_time, 1, 1, dim)
        dat_dict[str(s)] = dict(t=batch_t, x=x_reshaped)

    return dat_dict

