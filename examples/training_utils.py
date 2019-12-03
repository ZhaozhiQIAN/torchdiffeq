import time
import random

import torch
import torch.nn.functional as F


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


def get_batch(dat_dict, batch_size, seed=42):
    random.seed(seed)
    eids = random.sample(list(dat_dict.keys()), batch_size)
    return get_batch_from_eids(dat_dict, eids)


def get_all(dat_dict):
    eids = list(dat_dict.keys())
    return get_batch_from_eids(dat_dict, eids)


def training_loop(niters, data_fold, batch_size, optimizer, test_freq, loss_func, save_func):
    ii = 0
    best_loss = 10000

    start = time.time()
    for itr in range(1, niters + 1):

        t, y, y0, t_mask, y_mask, eids = get_batch(data_fold['train'], batch_size, itr * 7)

        optimizer.zero_grad()

        loss = loss_func(t, y, y0, t_mask, y_mask, eids)
        loss.backward()
        optimizer.step()

        if itr % test_freq == 0:
            with torch.no_grad():
                loss = loss_func(*get_all(data_fold['val']))
                print('Iter {:04d} | Total Loss {:.6f}'.format(itr, loss.item()))
                if loss < best_loss:
                    best_loss = loss
                    save_func()
                ii += 1

    end = time.time()
    return best_loss, end-start
