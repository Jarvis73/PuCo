import os
import time
import shutil
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import _LRScheduler

interpb = partial(F.interpolate, mode='bilinear', align_corners=True)
interpn = partial(F.interpolate, mode='nearest')


def get_palette(n_class):
    if n_class == 19:
        palette = [
            128, 64, 128, 
            244, 35, 232, 
            70, 70, 70, 
            102, 102, 156, 
            190, 153, 153, 
            153, 153, 153, 
            250, 170, 30,
            220, 220, 0, 
            107, 142, 35, 
            152, 251, 152, 
            70, 130, 180, 
            220, 20, 60, 
            255, 0, 0, 
            0, 0, 142, 
            0, 0, 70,
            0, 60, 100, 
            0, 80, 100, 
            0, 0, 230, 
            119, 11, 32]
    elif n_class == 16:
        palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
                    220, 220, 0, 107, 142, 35, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142,
                    0, 60, 100, 0, 0, 230, 119, 11, 32]
    else:
        raise ValueError

    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    cmap = np.array(palette).reshape(-1, 3)
    return palette, cmap


def get_label_mapper(dataset='gta5', n_class=19):
    if dataset.lower() == 'cityscapes':
        assert n_class == 19, f"`n_class` must be 16|19 for CityScapes"
        if n_class == 19:
            valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33,]
        else:
            valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 23, 24, 25, 26, 28, 32, 33,]
    elif dataset.lower() == 'gta5':
        assert n_class == 19, f"`n_class` must be 19 for GTA5"
        valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
    elif dataset.lower() == 'synthia':
        assert n_class == 16, f"`n_class` must be 16 for SYNTHIA"
        valid_classes = [3, 4, 2, 21, 5, 7, 15, 9, 6, 1, 10, 17, 8, 19, 12, 11,]
    else:
        raise ValueError(f"`dataset` can be cityscapes|gta5|synthia, got {dataset}")

    label2id = 250 * np.ones((35,), np.uint8)
    for i, l in enumerate(valid_classes):
        label2id[l] = i
    return label2id


def get_scheduler(optimizer, max_iter, gamma, warmup_lr=0, proj_lr_const=False):
    return PolynomialLR(optimizer, 
                        max_iter=max_iter, 
                        gamma=gamma,
                        warmup=warmup_lr, 
                        proj_lr_const=proj_lr_const)


class PolynomialLR(_LRScheduler):
    def __init__(self, optimizer, max_iter, decay_iter=1,
                 gamma=0.9, last_epoch=-1, warmup=0, proj_lr_const=False):
        self.decay_iter = decay_iter
        self.max_iter = max_iter
        self.gamma = gamma
        self.warmup = warmup
        self.proj_lr_const = proj_lr_const
        super(PolynomialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        i_iter = min(self.last_epoch, float(self.max_iter))
        if i_iter < self.warmup:
            factor = 0.1 + 0.9 * i_iter / self.warmup
        else:
            factor = (1 - i_iter / float(self.max_iter)) ** self.gamma
            factor = max(factor, 0)

        factors = [factor] * len(self.base_lrs)
        if self.proj_lr_const:
            factors[-1] = 1.0

        return [base_lr * factors[idx] for idx, base_lr in enumerate(self.base_lrs)]


def cross_entropy2d(inputs,
                    target,
                    weight=None,
                    softmax_used=False, # must be log_softmax
                    reduction='mean',
                    ignore_index=250):
    if len(target.size()) == 4:
        if softmax_used:
            loss = -torch.sum(target * inputs, dim=1)
        else:
            loss = -torch.sum(target * F.log_softmax(inputs), dim=1)
        if reduction == 'mean':
            return loss.mean()
        elif reduction == 'sum':
            return loss.sum()
        else:
            return loss

    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h != ht or w != wt:
        raise NotImplementedError(
            'sizes of input and label are not consistent')

    # input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    # target = target.view(-1)
    if softmax_used:
        loss = F.nll_loss(
            inputs, target, weight=weight, ignore_index=ignore_index,
            reduction=reduction)
    else:
        loss = F.cross_entropy(
            inputs, target, weight=weight, ignore_index=ignore_index,
            reduction=reduction)
    return loss


def save_ckpt(state_or_file, save_path=None, cloud=False):
    # remove old file first. AFS does not support overwrite file
    bk_path = None
    if cloud and save_path.exists():
        # rename-write-remove to avoid missing checkpoints when no space left
        bk_path = save_path.parent / (save_path.name + '.bk')
        os.rename(save_path, bk_path)
    try:
        if isinstance(state_or_file, dict):
            torch.save(state_or_file, save_path)
        else:
            shutil.copyfile(state_or_file, save_path)
    except Exception as e:
        if cloud and bk_path.exists():
            os.rename(bk_path, save_path)
        raise e
    if cloud and bk_path is not None and bk_path.exists():
        os.remove(bk_path)


def maybe_extract_datasets(opt, logger):
    logger.info(f"Extracting source dataset: {opt.src_dataset}")
    if opt.src_dataset == "gta5" and not Path("datasets/GTA5").exists():
        os.system('tar -xf afs/gta5.tar -C datasets')
    elif opt.src_dataset == "synthia" and not Path("datasets/SYNTHIA").exists():
        os.system('tar -xf afs/SYNTHIA.tar -C datasets')
    else:
        raise ValueError
    
    logger.info(f"Extracting target dataset: {opt.tgt_dataset}")
    if opt.tgt_dataset == "cityscapes" and not Path("datasets/CityScape").exists():
        os.system('tar -xf afs/CityScape.tar -C datasets')
    else:
        raise ValueError


class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.acc_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.spc = 0.           # Seconds per call
        self.cps = 0.           # Calls per second

        self.total_time = 0.    # Not affected by self.reset()
        self.total_calls = 0    # Not affected by self.reset()

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.acc_time += self.diff
        self.total_time += self.diff
        self.calls += 1
        self.total_calls += 1
        self.spc = self.acc_time / self.calls
        self.cps = self.calls / self.acc_time
        return self.diff

    def reset(self):
        self.acc_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.spc = 0.
        self.cps = 0.

    def start(self):
        return self

    def __enter__(self):
        self.tic()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.toc()


class Accumulator(object):
    def __init__(self, **kwargs):
        self.values = kwargs
        self.counter = {k: 0 for k, v in kwargs.items()}
        for k, v in self.values.items():
            if not isinstance(v, (float, int, list)):
                raise TypeError(f"The Accumulator does not support "
                                f"`{type(v)}`. Supported types: "
                                f"[float, int, list]")

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(self.values[k], list):
                self.values[k].append(v)
            else:
                self.values[k] = self.values[k] + v
            self.counter[k] += 1

    def reset(self):
        for k in self.values.keys():
            if isinstance(self.values[k], list):
                self.values[k] = []
            else:
                self.values[k] = 0
            self.counter[k] = 0

    def sum(self, key, axis=None):
        if isinstance(key, str):
            if isinstance(self.values[key], list):
                return np.array(self.values[key]).sum(axis)
            else:
                return self.values[key]
        else:
            return [self.sum(k, axis) for k in key]

    def mean(self, key, axis=None, dic=False):
        if isinstance(key, str):
            if isinstance(self.values[key], list):
                return np.array(self.values[key]).mean(axis)
            else:
                return self.values[key] / self.counter[key]
        elif isinstance(key, (list, tuple)):
            if dic:
                return {k: self.mean(k, axis) for k in key}
            return [self.mean(k, axis) for k in key]
        else:
            TypeError(f"`key` must be a str/list/tuple, got {type(key)}")

    def std(self, key, axis=None, dic=False):
        if isinstance(key, str):
            if isinstance(self.values[key], list):
                return np.array(self.values[key]).std(axis)
            else:
                raise RuntimeError("`std` is not supported for (int, float). "
                                   "Use list instead.")
        elif isinstance(key, (list, tuple)):
            if dic:
                return {k: self.std(k, axis) for k in key}
            return [self.std(k, axis) for k in key]
        else:
            TypeError(f"`key` must be a str/list/tuple, got {type(key)}")
