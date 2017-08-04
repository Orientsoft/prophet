import torch
import torch.nn as nn
import torch.optim as optim

from SampleLoader import SampleLoader
from SequenceLSTM import SequenceLSTM

if __name__ == "__init__":
    train_iteration_count = 100000
    input_feature_count = 19
    window_size = 12

    loader = SampleLoader(window_size)
    model = SequenceLSTM(input_feature_count, 100, 2, 1)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.paramteters(), lr=0.001, momentum=0.9)

    for i in range(train_iteration_count):
        # init
        data, label = loader.load()
        optimizer.zero_grad()

        # forward
        result = model(data)
        loss = criterion(result, label)

        # backward
        loss.backward()
        optimizer.step()
