import torch, math
from torch.optim.optimizer import Optimizer
import itertools as it
from .lookahead import *
from .ralamb import * 


def RangerLars(params, alpha=0.5, k=6, *args, **kwargs):
     ralamb = Ralamb(params, *args, **kwargs)
     return Lookahead(ralamb, alpha, k)
