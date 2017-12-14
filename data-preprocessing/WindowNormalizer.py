# -*- coding:utf-8 -*-

import numpy as np

def percentage_normalization(data):
    normalized_data = []

    p0 = data[0]
    normalized_data.append(0.)

    for i in range(1, len(data)):
        pi = data[i]
        ni = (pi / p0) - 1.
        normalized_data.append(ni)
    
    normalized_data = np.array(normalized_data)
    
    return normalized_data

def percentage_denormalization(p0, data):
    raw = np.zeros_like(data)

    for i in range(len(data)):
        ni = data[i]
        pi = p0 * (ni + 1.)
        raw[i] = pi

    return raw
