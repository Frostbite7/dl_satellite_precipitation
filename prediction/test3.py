import numpy as np

path = 'rain/'
path_test = 'rain_test/'

a = np.ones((2, 10, 10, 10, 2))
b = np.arange(1, 11)
b = np.reshape(b, [1, 10, 1, 1, 1])
c = a * b

for i in range(100):
    c.tofile(path + str(i) + '.bin')
    c.tofile(path_test + str(i) + '.bin')
