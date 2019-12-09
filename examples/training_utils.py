import time
import random
import pandas as pds

import torch
import torch.nn.functional as F

from config import DEVICE, D_TYPE


def get_data(path='data/cprd_full_4_markers.csv.gz', normalize=True):
    dat = pds.read_csv(path, compression='gzip')
    bio_markers = dat.columns[2:]

    if normalize:
        for b in bio_markers:
            dat[b] = (dat[b] - dat[b].mean()) / dat[b].std()
    dat_dict = dict()

    dat_grouped = dat.groupby('patid')

    for name, group in dat_grouped:
        t = group['ts'].values
        x = group[bio_markers].values
        len_t = len(t)
        dim = x.shape[1]
        x_reshaped = x.reshape(len_t, 1, 1, dim)
        dat_dict[str(name)] = dict(t=torch.tensor(t, dtype=D_TYPE), x=torch.tensor(x_reshaped, dtype=D_TYPE))
    return dat_dict


def get_fold(dat_dict, fold=5, seed=666):
    random.seed(seed)
    eids = list(dat_dict.keys())
    eid_set = set(eids)
    random.shuffle(eids)

    fold_list = list()
    for i in range(fold):
        eid_test = eids[i::fold]
        dat_test_fold = { k: dat_dict[k] for k in eid_test }

        eid_remain = list(eid_set - set(eid_test))
        eid_val = eid_remain[::10]
        dat_val_fold = { k: dat_dict[k] for k in eid_val }

        eid_train = list(set(eid_remain) - set(eid_val))

        dat_train_fold = { k: dat_dict[k] for k in eid_train }
        fold_dict = {'train': dat_train_fold, 'val': dat_val_fold, 'test': dat_test_fold}
        fold_list.append(fold_dict)

    return fold_list


def get_batch_from_eids(dat_dict, eids):
    t_list = [dat_dict[e]['t'] for e in eids]
    t_max = max([len(x) for x in t_list])
    t_padded = [F.pad(x, (0, t_max - len(x)), "constant", -1.).reshape((-1, 1)) for x in t_list]
    t_tensor = torch.cat(t_padded, dim=1)
    t_mask = t_tensor >= 0

    x_list = [dat_dict[e]['x'] for e in eids]
    x_padded = [F.pad(i, (0, 0, 0, 0, 0, 0, 0, t_max - i.shape[0]), "constant", -1.) for i in x_list]
    x_tensor = torch.cat(x_padded, dim=1)
    x_mask = x_tensor >= 0
    x0_tensor = x_tensor[0, ...]

    return t_tensor, x_tensor, x0_tensor, t_mask, x_mask, eids


def get_batch_from_eids_stack_t(dat_dict, eids):
    t_list = [dat_dict[e]['t'] for e in eids]
    x_list = [dat_dict[e]['x'] for e in eids]
    t_vec = torch.cat(t_list)
    t_tensor, _ = torch.sort(torch.unique(t_vec))
    t_dict = {str(t): i for i, t in enumerate(t_tensor.numpy())}
    x_tensor = torch.zeros((t_tensor.shape[0], len(eids), 1, x_list[0].shape[-1]))
    x_mask = torch.zeros_like(x_tensor, dtype=torch.bool)
    for m in range(len(eids)):
        tm = t_list[m].numpy()
        for i in range(len(tm)):
            ti = str(tm[i])
            ind = t_dict[ti]
            x_tensor[ind, m, 0, :] = x_list[m][i]
            x_mask[ind, m, 0, :] = ~torch.isnan(x_list[m][i])
    x_tensor[torch.isnan(x_tensor)] = 0.

    return t_tensor, x_tensor, x_mask, eids


def get_batch_stack_t(dat_dict, batch_size, seed=42):
    random.seed(seed)
    eids = random.sample(list(dat_dict.keys()), batch_size)
    return get_batch_from_eids_stack_t(dat_dict, eids)


def get_all_stack_t(dat_dict):
    eids = list(dat_dict.keys())
    return get_batch_from_eids_stack_t(dat_dict, eids)


def get_partition_stack_t(dat_dict, fold=50, seed=666):
    random.seed(seed)
    eids = list(dat_dict.keys())
    random.shuffle(eids)

    fold_list = list()
    for i in range(fold):
        eid_test = eids[i::fold]
        dat_test_fold = { k: dat_dict[k] for k in eid_test }

        fold_list.append(dat_test_fold)

    return fold_list


def get_input_for_grud(t, y, y_mask, encode_t, device=DEVICE):
    enc_step = int(torch.sum(t < encode_t))

    t_delta = t[1:] - t[:-1]
    t_delta = torch.cat((torch.zeros((1,)), t_delta))
    t_delta_mat = t_delta.reshape((-1, 1, 1, 1)).repeat((1, y.shape[1], 1, y.shape[-1]))
    last_y_mat = torch.cat((y[0:1, ...], y[:-1, ...]), dim=0)

    for i in range(1, t_delta_mat.shape[0]):
        last_delta = t_delta_mat[i-1, ...]
        last_mask = y_mask[i-1, ...].to(last_delta)
        t_delta_mat[i, ...] = t_delta_mat[i, ...] + last_delta * (1 - last_mask)
        last_y_mat[i, ...] = last_y_mat[i, ...] + last_y_mat[i-1, ...] * (1 - last_mask)

    y_target = y[enc_step:, ...].squeeze().to(device) # y is needed for loss
    y_target_mask = y_mask[enc_step:, ...].squeeze().to(device)
    gru_d_input = torch.cat((y, last_y_mat, y_mask.to(y), t_delta_mat), dim=2)[:enc_step, ...].permute((1, 2, 0, 3)).to(device)
    return y_target, y_target_mask, gru_d_input, t_delta[enc_step:].to(device)


def training_loop_v2(niters, data_fold, batch_size, encode_t, optimizer, test_freq, loss_func, save_func, device=DEVICE):
    ii = 0
    best_loss = 10000

    start = time.time()
    for itr in range(1, niters + 1):
        t, y, y_mask, eids = get_batch_stack_t(data_fold['train'], batch_size, itr * 7)
        y_target, y_target_mask, gru_d_input, t_delta_dec = get_input_for_grud(t, y, y_mask, encode_t, device)

        optimizer.zero_grad()

        loss = loss_func(y_target, y_target_mask, gru_d_input, t_delta_dec, 'train')
        loss.backward()
        optimizer.step()

        if itr % test_freq == 0:
            with torch.no_grad():
                val_fold_list = get_partition_stack_t(data_fold['val'], 50)
                total_loss = 0
                n = 0
                for f in val_fold_list:

                    t, y, y_mask, eids = get_all_stack_t(f)
                    y_target, y_target_mask, gru_d_input, t_delta_dec = get_input_for_grud(t, y, y_mask, encode_t, device)

                    loss = loss_func(y_target, y_target_mask, gru_d_input, t_delta_dec, 'val')
                    this_n = torch.sum(y_target_mask.to(y)).numpy()
                    n += this_n
                    total_loss += loss.item() * this_n
                loss = total_loss / n
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss))
                if loss < best_loss:
                    best_loss = loss
                    save_func()
                ii += 1

    end = time.time()
    return best_loss, end-start


# @deprecated
def training_loop(niters, data_fold, batch_size, optimizer, test_freq, loss_func, save_func, batch_method='non-stack'):
    ii = 0
    best_loss = 10000

    start = time.time()
    for itr in range(1, niters + 1):
        if batch_method == 'non-stack':
            t, y, y0, t_mask, y_mask, eids = get_batch(data_fold['train'], batch_size, itr * 7)
        else:
            t, y, y0, t_mask, y_mask, eids = get_batch_stack_t(data_fold['train'], batch_size, itr * 7)

        optimizer.zero_grad()

        loss = loss_func(t, y, y0, t_mask, y_mask, eids)
        loss.backward()
        optimizer.step()

        if itr % test_freq == 0:
            with torch.no_grad():
                if batch_method == 'non-stack':
                    loss = loss_func(*get_all(data_fold['val']))
                else:
                    raise ValueError('This function can only run in non-stack mode.')
                    # loss = loss_func(*get_batch_stack_t(data_fold['val'], batch_size))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                if loss < best_loss:
                    best_loss = loss
                    save_func()
                ii += 1

    end = time.time()
    return best_loss, end-start


# @deprecated
def get_batch(dat_dict, batch_size, seed=42):
    random.seed(seed)
    eids = random.sample(list(dat_dict.keys()), batch_size)
    return get_batch_from_eids(dat_dict, eids)


# @deprecated
def get_all(dat_dict):
    eids = list(dat_dict.keys())
    return get_batch_from_eids(dat_dict, eids)
