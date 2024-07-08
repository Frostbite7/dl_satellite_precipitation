import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def pdf(x, a, h):
    y = tf.exp((-(x - a) ** 2) / (2 * h ** 2)) / (tf.sqrt(2 * np.pi) * h)
    y = tf.reduce_mean(y)
    return y


def num_kl_div(a, b, h, min, max, n_bins):
    div = 0.0
    step = (max-min) / n_bins
    x = step / 2 + min
    for i in range(n_bins):
        temp = pdf(x, a, h) * tf.log(pdf(x, a, h) / pdf(x, b, h))
        div = div + step * pdf(x, a, h) * tf.log(pdf(x, a, h) / tf.maximum(pdf(x, b, h), 1e-35))
        # print(pdf(x, a, h).eval())
        # print(pdf(x, b, h).eval())
        # print(tf.log(pdf(x, a, h)/pdf(x, b, h)).eval())
        x = x + step
    return div


def com_mse(a, b):
    r = (a-b)*(a-b)
    return tf.reduce_mean(r)


h = 2.5
# a = tf.constant(np.array([i for i in range(50)]), tf.float32)
a = tf.constant(np.fromfile('test_data/rain_value_list.bin'), dtype=tf.float32)
n = a.shape.as_list()[0]
b = tf.constant([0.0] * n, dtype=tf.float32)

# v = tf.get_variable('v', (), initializer=tf.constant_initializer(0.1))
# c = b + v
div_n = num_kl_div(a, b, h, -10.0, 50.0, 30)
mse = com_mse(a, b)
# train = tf.train.AdamOptimizer(0.1).minimize(mse)

sess = tf.InteractiveSession()
# sess.run(tf.global_variables_initializer())
print(div_n.eval(), mse.eval())

# sess.run(tf.global_variables_initializer())
# print(sess.run(v))
# sess.run(train)
# print(sess.run(v))

# test = pdf(0, b, h)
# print(test.eval())

# x = np.linspace(-5, 70, 1000)
# y = np.array([pdf(i, a, h).eval() for i in x])
# plt.plot(x, y)
# plt.show()