import os
import time
import shutil
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn

import torch
import numpy as np
from torch.optim import SGD, Adam
from tensorboardX import SummaryWriter

class Averager():
    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():
    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer

def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape): 
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n) 
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def to_pixel_samples(img):
    """ Convert the image to coord-Gray pairs.
        img: Tensor, (1, H, W)
    """

    coord = make_coord(img.shape[-2:])
    rgb = img.view(1, -1).permute(1, 0) 
    return coord, rgb    


def make_coord_(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    B, _, H, W = shape 
    coord_seqs = []
    for i, n in enumerate([H, W]): 
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1) 
    ret = ret.unsqueeze(0).expand(B, H, W, 2)
    if flatten:
        ret = ret.view(B, -1, ret.shape[-1]) 
    return ret


def to_pixel_samples_(img):
    """ Convert the image to coord-Gray pairs.
    """
    B, _, H, W = img.shape  
    coord = make_coord_(img.shape) 
    rgb = img.permute(0, 2, 3, 1).contiguous().view(B, -1, 1) 
    return coord, rgb


def get_rgb_from_coord(img, coord, coord_sample):
    """ Get the RGB values corresponding to the given coordinates.
    Args:
        img (torch.Tensor): Input image tensor of shape [B, 1, H, W].
        coord (torch.Tensor): Coordinate tensor of shape [B, H*W, 2].
        coord_sample (torch.Tensor): Coordinate tensor of shape [B, M, 2].
    Returns:
        torch.Tensor: RGB values corresponding to the given coordinates, of shape [B, M, 1].
    """
    B, _, H, W = img.shape
    M = coord_sample.shape[1]  
    img = img.view(B, -1, 1)  
    coord = coord.view(B, H*W, 2)  
    coord_sample = coord_sample.view(B, M, 2)  
    
    _, indices = torch.max((coord.unsqueeze(1) == coord_sample.unsqueeze(2)).all(3), dim=2)
    indices = indices.unsqueeze(2)
    indices = indices.expand(B, M, 1)
    
    rgb = torch.gather(img.unsqueeze(2), dim=1, index=indices.unsqueeze(2))
    
    return rgb


def calc_psnr(sr, hr, dataset=None, scale=1, rgb_range=1):
    diff = (sr - hr) / rgb_range
    if dataset is not None:
        if dataset == 'benchmark':
            shave = scale
            if diff.size(1) > 1:
                gray_coeffs = [65.738, 129.057, 25.064]
                convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
                diff = diff.mul(convert).sum(dim=1)
        elif dataset == 'div2k':
            shave = scale + 6
        else:
            raise NotImplementedError
        valid = diff[..., shave:-shave, shave:-shave]
    else:
        valid = diff
    mse = valid.pow(2).mean()
    return -10 * torch.log10(mse)


def smooth_loss(flow):
    assert flow.dim() == 4 or flow.dim() == 5, 'Smooth_loss: dims match failed.'

    if flow.dim() == 5:
        dy = torch.abs(flow[:, :, 1:, :, :] - flow[:, :, :-1, :, :])
        dx = torch.abs(flow[:, :, :, 1:, :] - flow[:, :, :, :-1, :])
        dz = torch.abs(flow[:, :, :, :, 1:] - flow[:, :, :, :, :-1])

        dy = dy * dy
        dx = dx * dx
        dz = dz * dz

        d = torch.mean(dx) + torch.mean(dy) + torch.mean(dz)
    else:
        dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
        dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])

        dx = dx * dx
        dy = dy * dy
        d = torch.mean(dx) + torch.mean(dy)

    grad = d / 3.0
    return grad


class Transformer2D(nn.Module):
    def __init__(self):
        super(Transformer2D, self).__init__()

    def forward(self, src, flow, padding_mode="border"):
        b = flow.shape[0]
        size = flow.shape[2:]
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = grid.to(torch.float32)
        grid = grid.repeat(b, 1, 1, 1).to(flow.device)
        new_locs = grid + flow
        shape = flow.shape[2:]
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1, 0]]
        warped = F.grid_sample(src, new_locs, align_corners=True, padding_mode=padding_mode)
        return warped
