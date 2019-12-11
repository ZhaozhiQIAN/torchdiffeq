import pandas as pds
import numpy as np
import time
import importlib
import random
import argparse
import pickle

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


parser = argparse.ArgumentParser('ODE Enc Dec')
parser.add_argument('--mode', type=str, choices=['train', 'test'], default='train')
parser.add_argument('--method', type=str, choices=['grud', 'ode', 'attn'], default='ode')
parser.add_argument('--encode_t', type=str, choices=['1', '2', '3'], default='1')
parser.add_argument('--ode', type=str, choices=['standard', 'fs', 'ho', 'linear'], default='standard')
parser.add_argument('--ode_solver', type=str, choices=['adjoint', 'dto'], default='adjoint')
parser.add_argument('--fresh_train', type=str, choices=['yes', 'no'], default='yes')
parser.add_argument('--gpu', type=str, choices=['0', '1'], default='0')
args = parser.parse_args()
DEVICE = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')


class GrudEncDec():
    def run(self, mode, fresh_train, encode_t):
        # data
        dat_dict = training_utils.get_data('data/cprd_sample_5_obs_6yr.csv.gz', normalize=False)
        dat_folds = training_utils.get_fold(dat_dict, fold=5, seed=666)

        # parameters
        model_path = 'exp2-models/grud-enc-dec-{}.pth'
        dim_y = 9
        dim_hidden = 50
        dim_decoder_mlp_hidden = 50

        niters = 5000
        test_freq = 100
        data_fold = dat_folds[0]
        batch_size = 100
        encode_t = float(encode_t)

        # models
        gru = GRUD.GRUD_V2(input_size=dim_y, hidden_size=50, output_last=True, device=DEVICE)
        seq_decoder = baseline_models.BaselineTimeGRU(dim_hidden, dim_hidden, dim_hidden).to(DEVICE)
        decoder = baseline_models.SLP(dim_hidden, dim_y, dim_decoder_mlp_hidden).to(DEVICE)

        def grud_enc_dec_loss_func(y_target, y_target_mask, gru_d_input, t_delta_dec, mode='train'):
            res = gru(gru_d_input)
            t_delta_dec_mat = t_delta_dec.reshape((-1, 1)).repeat((1, y_target.shape[1])).to(DEVICE)
            y_pred = seq_decoder(res, t_delta_dec_mat)
            y_out = decoder(y_pred)
            loss = torch.mean(torch.abs(y_out[y_target_mask] - y_target[y_target_mask]))
            return loss

        def grud_enc_dec_save_func():
            torch.save(gru.state_dict(), model_path.format('encoder'))
            torch.save(seq_decoder.state_dict(), model_path.format('seq-decoder'))
            torch.save(decoder.state_dict(), model_path.format('decoder'))

        def grud_enc_dec_load_func():
            gru.load_state_dict(torch.load(model_path.format('encoder')))
            gru.eval()
            seq_decoder.load_state_dict(torch.load(model_path.format('seq-decoder')))
            seq_decoder.eval()
            decoder.load_state_dict(torch.load(model_path.format('decoder')))
            decoder.eval()

        if fresh_train != 'yes' or mode != 'train':
            grud_enc_dec_load_func()

        optimizer = optim.Adam(list(gru.parameters()) +
                               list(seq_decoder.parameters()) +
                               list(decoder.parameters()), lr=1e-3)

        if mode == 'train':
            out = training_utils.training_loop_v2(niters,
                                                  data_fold,
                                                  batch_size,
                                                  encode_t,
                                                  optimizer,
                                                  test_freq,
                                                  grud_enc_dec_loss_func,
                                                  grud_enc_dec_save_func,
                                                  DEVICE)
        else:
            out = training_utils.testing(data_fold, encode_t, grud_enc_dec_load_func, DEVICE)
        print(out)
        return out


class OdeDecGrudEnc():
    def run(self, mode, fresh_train='yes', ode_type='standard', ode_solver='adjoint', encode_t='1'):
        print('Training OdeDecGrudEnc')
        # data
        dat_dict = training_utils.get_data('data/cprd_sample_5_obs_6yr.csv.gz', normalize=False)
        dat_folds = training_utils.get_fold(dat_dict, fold=5, seed=666)

        # parameters
        spec_string = '{}_{}_{}'.format(ode_type, ode_solver, encode_t)
        model_path = 'exp2-models/ode_dec_grud_enc_' + spec_string + '_{}.pth'
        dim_y = 9
        dim_hidden = 50
        dim_decoder_mlp_hidden = 50

        niters = 2000
        test_freq = 50
        data_fold = dat_folds[0]
        batch_size = 100
        encode_t = float(encode_t)

        # models
        gru = GRUD.GRUD_V2(input_size=dim_y, hidden_size=50, output_last=True, device=DEVICE)
        if ode_type == 'standard':
            ode_func = ode_models.ODEFunc0(dim_hidden, dim_hidden).to(DEVICE)
        elif ode_type == 'fs':
            ode_func = ode_models.FSODE(dim_hidden, dim_hidden).to(DEVICE)
        elif ode_type == 'ho':
            ode_func = ode_models.HigherOrderOdeLatent(dim_hidden, dim_hidden).to(DEVICE)
        elif ode_type == 'linear':
            ode_func = ode_models.ODELinear(dim_hidden).to(DEVICE)
        else:
            raise ValueError('Unsupported ode type')
        decoder = baseline_models.SLP(dim_hidden, dim_y, dim_decoder_mlp_hidden).to(DEVICE)

        def ode_dec_grud_enc_loss_func(y_target, y_target_mask, gru_d_input, t_delta_dec, mode='train'):
            if mode == 'train':
                ode_func.reset_counter()
            init_cond = gru(gru_d_input)[:, None, :]
            t_dec = torch.cumsum(t_delta_dec, 0)
            if ode_solver == 'adjoint':
                y_pred = odeint(ode_func, init_cond, t_dec)
            elif ode_solver == 'dto':
                y_pred = dto(ode_func, init_cond, t_dec)
            else:
                raise ValueError('Incorrect ode solver')
            if mode == 'train':
                ode_func.reset_counter()
            y_out = decoder(y_pred.squeeze())
            loss = torch.mean(torch.abs(y_out[y_target_mask] - y_target[y_target_mask]))
            return loss

        def ode_dec_grud_enc_save_func():
            torch.save(gru.state_dict(), model_path.format('encoder'))
            torch.save(ode_func.state_dict(), model_path.format('ode_func'))
            torch.save(decoder.state_dict(), model_path.format('decoder'))
            with open('exp2-models/ode_dec_grud_enc_' + ode_type + '_counter.pkl', 'wb') as f:
                pickle.dump(ode_func.counter_list, f)

        def ode_dec_grud_enc_load_func():
            gru.load_state_dict(torch.load(model_path.format('encoder')))
            gru.eval()
            ode_func.load_state_dict(torch.load(model_path.format('ode_func')))
            ode_func.eval()
            decoder.load_state_dict(torch.load(model_path.format('decoder')))
            decoder.eval()

        if fresh_train != 'yes' or mode != 'train':
            ode_dec_grud_enc_load_func()

        optimizer = optim.Adam(list(gru.parameters()) +
                               list(ode_func.parameters()) +
                               list(decoder.parameters()), lr=1e-3)

        if mode == 'train':
            out = training_utils.training_loop_v2(niters,
                                                  data_fold,
                                                  batch_size,
                                                  encode_t,
                                                  optimizer,
                                                  test_freq,
                                                  ode_dec_grud_enc_loss_func,
                                                  ode_dec_grud_enc_save_func,
                                                  DEVICE)
        else:
            out = training_utils.testing(data_fold, encode_t, ode_dec_grud_enc_loss_func, DEVICE)
        print(out)
        return out


class OdeDecAttn():
    def run(self, mode, fresh_train='yes', ode_type='standard', encode_t='1'):
        print('{} OdeDecAttn'.format(mode))
        # data
        dat_dict = training_utils.get_data('data/cprd_sample_5_obs_6yr.csv.gz', normalize=False)
        dat_folds = training_utils.get_fold(dat_dict, fold=5, seed=666)

        # parameters
        spec_string = '{}_{}'.format(ode_type, encode_t)
        model_path = 'exp2-models/ode_dec_attn_' + spec_string + '_{}.pth'
        dim_y = 9
        dim_hidden = 50
        dim_hidden_ode = 50 if ode_type == 'standard' else None
        dim_decoder_mlp_hidden = 50

        niters = 2000
        test_freq = 50
        data_fold = dat_folds[0]
        batch_size = 100
        encode_t = float(encode_t)

        # models
        gru = GRUD.GRUD_V2(input_size=dim_y, hidden_size=50, output_last=False, device=DEVICE)
        ode_func = ode_models.AttentiveODE(dim_hidden, dim_hidden_ode).to(DEVICE)
        decoder = baseline_models.SLP(dim_hidden, dim_y, dim_decoder_mlp_hidden).to(DEVICE)

        def ode_dec_attn_loss_func(y_target, y_target_mask, gru_d_input, t_delta_dec, mode='train'):
            if mode == 'train':
                ode_func.reset_counter()

            gru_hidden = gru(gru_d_input)
            init_cond = gru_hidden[:, -2:-1, :]
            t_dec = torch.cumsum(t_delta_dec, 0)

            ode_func.set_gru_hidden(gru_hidden)
            y_pred = dto(ode_func, init_cond, t_dec)

            if mode == 'train':
                ode_func.reset_counter()
            y_out = decoder(y_pred.squeeze())
            loss = torch.mean(torch.abs(y_out[y_target_mask] - y_target[y_target_mask]))
            return loss

        def ode_dec_attn_save_func():
            torch.save(gru.state_dict(), model_path.format('encoder'))
            torch.save(ode_func.state_dict(), model_path.format('ode_func'))
            torch.save(decoder.state_dict(), model_path.format('decoder'))
            with open('exp2-models/ode_dec_attn_' + ode_type + '_counter.pkl', 'wb') as f:
                pickle.dump(ode_func.counter_list, f)

        def ode_dec_attn_load_func():
            gru.load_state_dict(torch.load(model_path.format('encoder')))
            gru.eval()
            ode_func.load_state_dict(torch.load(model_path.format('ode_func')))
            ode_func.eval()
            decoder.load_state_dict(torch.load(model_path.format('decoder')))
            decoder.eval()

        if fresh_train != 'yes' or mode != 'train':
            ode_dec_attn_load_func()

        optimizer = optim.Adam(list(gru.parameters()) +
                               list(ode_func.parameters()) +
                               list(decoder.parameters()), lr=1e-3)

        if mode == 'train':
            out = training_utils.training_loop_v2(niters,
                                                  data_fold,
                                                  batch_size,
                                                  encode_t,
                                                  optimizer,
                                                  test_freq,
                                                  ode_dec_attn_loss_func,
                                                  ode_dec_attn_save_func,
                                                  DEVICE)
        else:
            out = training_utils.testing(data_fold, encode_t, ode_dec_attn_loss_func, DEVICE)
        print(out)
        return out


if __name__ == '__main__':
    if args.method == 'grud':
        GrudEncDec().run(args.mode, args.fresh_train, args.encode_t)
    elif args.method == 'ode':
        OdeDecGrudEnc().run(args.mode, args.fresh_train, args.ode, args.ode_solver, args.encode_t)
    elif args.method == 'attn':
        OdeDecAttn().run(args.mode, args.fresh_train, args.ode, args.encode_t)
    else:
        raise ValueError
