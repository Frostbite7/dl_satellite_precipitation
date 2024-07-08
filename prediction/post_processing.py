import numpy as np
import tensorflow as tf


def results(y, y_):
    y = np.array(y)
    y = np.transpose(y, (1, 0, 2, 3, 4))
    y = y[:, :, :, :, 0]
    y_ = y_[:, :, :, :, 0]

    print('y:', y[0, :, 1, 1])
    print('y_:', y_[0, :, 1, 1])

    epsilon = 10 ** (-9)
    correlate = np.sum(y * y_, (2, 3)) / (np.sqrt(np.sum(y * y, (2, 3)) * np.sum(y_ * y_, (2, 3))) + epsilon)
    correlation = np.mean(correlate, 0)

    bin_y = np.where(y > 0.5, 1, 0)
    bin_y_ = np.where(y_ > 0.5, 1, 0)

    hits = np.sum(np.where((bin_y == 1) & (bin_y_ == 1), 1, 0), (0, 2, 3))
    misses = np.sum(np.where((bin_y == 0) & (bin_y_ == 1), 1, 0), (0, 2, 3))
    falsealarms = np.sum(np.where((bin_y == 1) & (bin_y_ == 0), 1, 0), (0, 2, 3))

    csi = hits / (hits + misses + falsealarms)
    far = falsealarms / (hits + falsealarms)
    pod = hits / (hits + misses)

    return correlation, csi, far, pod
