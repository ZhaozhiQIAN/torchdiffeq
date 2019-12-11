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

    true_y0 = torch.tensor([[0.0, 5.0]])
    t = torch.linspace(0, 5, args.data_size + 1)

    class Lambda(nn.Module):
        def __init__(self, w0, f0, w1, dr):
            super(Lambda, self).__init__()

            self.w0 = torch.tensor(w0)
            self.f0 = torch.tensor(f0)
            self.w1 = torch.tensor(w1)
            self.dr = torch.tensor(dr)

        def force(self, t):
            return self.f0 * torch.sin(self.w1 * t)

        def forward(self, t, y):
            dy0_dt = y[:, 1]
            dy1_dt = -2. * self.w0 * y[:, 1] * self.dr - self.w0 ** 2 * y[:, 0] + self.force(t)
            return torch.cat((dy0_dt.reshape(-1, 1), dy1_dt.reshape(-1, 1)), axis=1)

    # This numerical solution given the true DE.
    with torch.no_grad():
        lam = Lambda(5., 5., 3., 0.01)
        true_y = odeint(lam, true_y0, t, method='dopri5')

    dat_dict = dict()

    for s in range(args.data_size - args.batch_time):
        batch_t = t[s:s+args.batch_time]
        batch_y = torch.stack([true_y[s + i] for i in range(args.batch_time)], dim=0)
        dim = batch_y.shape[-1]
        x_reshaped = batch_y.reshape(args.batch_time, 1, 1, dim)
        dat_dict[str(s)] = dict(t=batch_t, x=x_reshaped)

    return dat_dict

