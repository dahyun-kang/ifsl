r""" Helper functions """
import random

import torch
import numpy as np


def fix_randseed(seed):
    r""" Set random seeds for reproducibility """
    if seed is None:
        seed = int(random.random() * 1e5)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def to_cpu(tensor):
    return tensor.detach().clone().cpu()


def print_param_count(model):
    backbone_param = 0
    learner_param = 0
    for k in model.state_dict().keys():
        n_param = model.state_dict()[k].view(-1).size(0)
        if k.split('.')[0] in 'backbone':
            if k.split('.')[1] in ['classifier', 'fc']:  # as fc layers are not used in HSNet/ASNet
                continue
            backbone_param += n_param
        else:
            learner_param += n_param

    msg = f'Backbone # param.: {backbone_param:,}\n'
    msg += f'Learnable # param.: {learner_param:,}\n'
    msg += f'Total # param.: {backbone_param + learner_param:,}\n'
    print(msg)
