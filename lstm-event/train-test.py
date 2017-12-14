# -*- coding:utf-8 -*-

import sys
sys.path.append("../utility")
from Display import Display

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

import os
import numpy as np

from SampleLoader import SampleLoader
from SequenceLSTM import SequenceLSTM

def check_test(train_iter, test_interval):
    if (train_iter != 0) and (train_iter % test_interval == 0):
        return True
    else:
        return False

def check_snapshot(train_iter, snapshot_interval):
    if (train_iter != 0) and (train_iter % snapshot_interval == 0):
        return True
    else:
        return False

if __name__ == "__main__":
    # parameter
    train_iteration_count = 100000000

    input_feature_count = 6
    output_feature_count = 4
    batch_size = 4
    window_size = 30

    test_interval = 500
    test_iteration = 5

    snapshot_interval = 500
    snapshot_sub_path = "snapshot/"
    snapshot_prefix = "pt_lstm"
    snapshot = ""
    
    display_interval = 10

    # setup
    display = Display(display_interval)

    loader = SampleLoader(db_name="tploader", collection_prefix="tploader-1m", test_index_offset=39600, batch_size=batch_size, data_length=window_size)
    loader.start_load_train()
    loader.start_load_test()

    net = SequenceLSTM(input_feature_count, 128, 2, output_feature_count, batch_size, window_size)
    print(net)

    last_train_iter = 0
    # load snapshot if needed
    if snapshot != "":
        net = torch.load("./{0}{1}".format(snapshot_sub_path, snapshot))
        name = snapshot.split('.')[1]
        fields = name.split('_')
        last_train_iter = int(fields[len(fields) - 1])

    net = net.float()
    gpu_flag = torch.cuda.is_available()
    if gpu_flag:
        net = net.cuda()

    criterion = nn.MSELoss()
    optimizer = optim.SGD(net.parameters(), lr=0.0000001, momentum=0.9)

    train_loss = np.zeros(train_iteration_count - last_train_iter)
    test_loss = np.zeros(train_iteration_count - last_train_iter)
    train_iter = 0

    # training loop
    for train_iter in range(last_train_iter, train_iteration_count):
        # init
        train_iter_offset = train_iter - last_train_iter

        data, label = loader.load_train_sample()
        data, label = np.array(data), np.array(label)
        if gpu_flag:
            data, label = torch.cuda.FloatTensor(data), torch.cuda.FloatTensor(label)
            # print(data)
        else:
            data, label = torch.from_numpy(data), torch.from_numpy(label)
        data, label = Variable(data, requires_grad=True), Variable(label)
        
        net.train(True)
        optimizer.zero_grad()

        # forward
        result = net(data)
        label = np.squeeze(label)
        loss = criterion(result, label)

        # backward
        loss.backward()
        optimizer.step()

        # update train loss
        loss = loss.cpu()
        train_loss[train_iter_offset] = loss.data.numpy()[0]
        if loss.data.numpy()[0] > 100.:
            print("loss: {0}\nresult: {1}\nlabel: {2}".format(loss, result, label))

        # test if needed
        if check_test(train_iter, test_interval):
            avg_loss = 0.

            for test_iter in range(test_iteration):
                # init
                data, label = loader.load_test_sample()
                data, label = np.array(data), np.array(label)
                if gpu_flag:
                    data, label = torch.cuda.FloatTensor(data), torch.cuda.FloatTensor(label)
                    # print(data)
                else:
                    data, label = torch.from_numpy(data), torch.from_numpy(label)
                data, label = Variable(data, requires_grad=True), Variable(label)

                net.train(False)
                optimizer.zero_grad()

                # forward
                result = net(data)
                loss = criterion(result, label)

                # update loss
                loss = loss.cpu()
                avg_loss += loss.data.numpy()[0]

            avg_loss /= test_iteration
            test_loss[train_iter_offset] = avg_loss
        else:
            test_loss[train_iter_offset] = test_loss[train_iter - last_train_iter - 1]

        # snapshot if needed
        if check_snapshot(train_iter, snapshot_interval):
            snapshot_path = "./" + snapshot_sub_path

            # init path
            if not os.path.isdir(snapshot_path):
                os.makedirs(snapshot_path)

            # save model state
            snapshot_state_file = "{0}{1}_{2}_{3}.{4}".format(snapshot_path, snapshot_prefix, "iter", train_iter, "state")
            torch.save(net, snapshot_state_file)

            # save network weight
            snapshot_weight_file = "{0}{1}_{2}_{3}.{4}".format(snapshot_path, snapshot_prefix, "iter", train_iter, "weight")
            torch.save(net.state_dict(), snapshot_weight_file)

        # print status
        display.update(train_iter, loss.data.numpy()[0])

        # TODO : plot loss