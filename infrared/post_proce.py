import numpy as np

def results(y_conv, y):
    y_conv = np.where(y_conv<0, 0, 1)

    hits = np.sum(np.where((y_conv == 1) & (y == 1), 1, 0))
    misses = np.sum(np.where((y_conv == 0) & (y == 1), 1, 0))
    falsealarms = np.sum(np.where((y_conv == 1) & (y == 0), 1, 0))

    csi = hits / (hits + misses + falsealarms)
    far = falsealarms / (hits + falsealarms)
    pod = hits / (hits + misses)

    return csi, pod, far