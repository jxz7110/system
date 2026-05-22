from typing import Callable, Optional
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
import math
import numpy as np
from layers.model_layers import *
from layers.mixF_layers import *
from layers.chain_mixF import *
from layers.MixF3 import easyBackbone3
class Model(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.model = easyBackbone3(configs)

    def forward(self,x):
        x = self.model(x)
        return x

