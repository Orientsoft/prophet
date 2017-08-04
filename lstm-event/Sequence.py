import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class SequenceLSTM(nn.Module):
    def __init__(self, baby_sitter):
        super(Sequence, self).__init__()
        self.baby_sitter = baby_sitter
    
    def forward(self, input, future=0):
