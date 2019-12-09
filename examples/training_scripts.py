import pandas as pds
import numpy as np
import time
import importlib
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchdiffeq import odeint_adjoint as odeint
from torchdiffeq import odeint as dto

import ode_models
import baseline_models
import training_utils
import GRUD
from config import DEVICE, D_TYPE


class GrudEncDec():
    def run(self):
        # data
        dat_dict = training_utils.get_data('data/cprd_sample_5_obs_6yr.csv.gz', normalize=False)
        dat_folds = training_utils.get_fold(dat_dict, fold=5, seed=666)

        # parameters
        dim_y = 9
        dim_hidden = 50
        dim_decoder_mlp_hidden = 50

        niters = 5000
        test_freq = 100
        data_fold = dat_folds[0]
        batch_size = 100
        encode_t = 1.

        # models
        gru = GRUD.GRUD(input_size=dim_y, hidden_size=50, output_last=True)
        seq_decoder = baseline_models.BaselineTimeGRU(dim_hidden, dim_hidden, dim_hidden).to(DEVICE)
        decoder = baseline_models.SLP(dim_hidden, dim_y, dim_decoder_mlp_hidden).to(DEVICE)

        def grud_enc_dec_loss_func(y_target, y_target_mask, gru_d_input, t_delta_dec):
            res = gru(gru_d_input)
            t_delta_dec_mat = t_delta_dec.reshape((-1, 1)).repeat((1, y_target.shape[1])).to(DEVICE)
            y_pred = seq_decoder(res, t_delta_dec_mat)
            y_out = decoder(y_pred)
            loss = torch.mean(torch.abs(y_out[y_target_mask] - y_target[y_target_mask]))
            return loss

        def grud_enc_dec_save_func():
            model_path = 'exp2-models/grud-enc-dec-{}.pth'
            torch.save(gru.state_dict(), model_path.format('encoder'))
            torch.save(seq_decoder.state_dict(), model_path.format('seq-decoder'))
            torch.save(decoder.state_dict(), model_path.format('decoder'))

        optimizer = optim.Adam(list(gru.parameters()) +
                               list(seq_decoder.parameters()) +
                               list(decoder.parameters()), lr=1e-3)

        out = training_utils.training_loop_v2(niters,
                                              data_fold,
                                              batch_size,
                                              encode_t,
                                              optimizer,
                                              test_freq,
                                              grud_enc_dec_loss_func,
                                              grud_enc_dec_save_func,
                                              DEVICE)
        print(out)
        return out


if __name__ == '__main__':
    GrudEncDec().run()
