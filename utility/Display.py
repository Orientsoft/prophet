# -*- coding:utf-8 -*-

import time
import numpy as np

class Display(object):
    def __init__(self, display_interval=10):
        self.display_interval = display_interval
        self.count = 0
        self.train_loss = []
        self.last_print_ts = time.time()

    def print_loss(self, iter, last_loss, avg_loss, period):
        speed = self.display_interval / period
        print("Iteration {0} - loss: {1}, avg_loss: {2}\nSpeed: {3}iter/s, {4}s/{5}iter(s)\n".format(iter, last_loss, avg_loss, speed, period, self.display_interval))

    def update(self, iter, loss):
        self.train_loss.append(loss)

        self.count += 1
        if self.count % self.display_interval == 0:
            loss_array = np.array(self.train_loss)
            last_loss = loss_array[-1]
            avg_loss = np.average(loss_array)

            ts = time.time()
            period = ts - self.last_print_ts
            self.print_loss(iter, last_loss, avg_loss, period)

            self.last_print_ts = ts
            self.train_loss.clear()