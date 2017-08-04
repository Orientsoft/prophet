# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class SequenceLSTM(nn.Module):
    def __init__(self, input_feature_count, hidden_size, layer_count, output_feature_count):
        super(SequenceLSTM, self).__init__()

        self.input_feature_count = input_feature_count
        self.hidden_size = hidden_size
        self.layer_count = layer_count
        self.output_feature_count = output_feature_count

        self.seq_lstm = nn.LSTM(input_feature_count, hidden_size, layer_count, batch_first=True)
        self.linear = nn.Linear(input_feature_count, output_feature_count)

    def forward(self, data):
        result = self.seq_lstm(data)
        
        # get last data (prediction) in sequence
        result = result[:, -1, :]

        result = self.linear(result)

        return result
