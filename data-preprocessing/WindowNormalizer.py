# -*- coding:utf-8 -*-

import numpy as np

def percentage_normalization(self, data, label):
    normalized_data = np.zeros_like(data)
    normalized_label = np.zeros_like(label)

    p0 = data[0]
    normalized_data[0] = 0.

    for i in range(1, len(data)):
        pi = data[i]
        ni = (pi / p0) - 1.
        normalized_data[i] = ni

    for i in range(len(label)):
        pi = label[i]
        ni = (pi / p0) - 1.
        normalized_label[i] = ni

    return normalized_data, normalized_label

def percentage_denormalization(self, p0, data):
    raw = np.zeros_like(data)

    for i in range(len(data)):
        ni = data[i]
        pi = p0 * (ni + 1.)
        raw[i] = pi

    return raw
