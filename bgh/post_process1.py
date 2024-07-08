import tensorflow as tf
import numpy as np

ds = tf.contrib.distributions
epsilon = 10 ** (-35)


def k_l_divergence(y_true, y_pred):
    y_true = tf.clip_by_value(y_true, epsilon, 1)
    y_pred = tf.clip_by_value(y_pred, epsilon, 1)
    return tf.reduce_sum(y_true * tf.log(y_true / y_pred), axis=-1)


def kl_div(prediction, ground, maxi, n_bins):
    hist_p = tf.histogram_fixed_width(prediction, [0.0, maxi], nbins=n_bins, dtype=tf.float32)
    hist_g = tf.histogram_fixed_width(ground, [0.0, maxi], nbins=n_bins, dtype=tf.float32)
    batch_size = prediction.get_shape().as_list()[0]
    dist_p = hist_p / batch_size
    dist_g = hist_g / batch_size
    div = k_l_divergence(dist_g, dist_p)
    return div


# a = tf.constant([i+0.0 for i in range(50)] + [1.0]*500)
a = tf.constant(np.fromfile('test_data/rain_value_list.bin'), dtype=tf.float32)
b = tf.constant([0.0]*2208)
# v = tf.get_variable('v', [1], initializer=tf.constant_initializer(0))
# c = a * v
loss = kl_div(a, b, 50, 20)
# loss = tf.reduce_sum((c-b)*(c-b),axis=-1)
# train = tf.train.AdamOptimizer(0.1).minimize(loss)
# d = tf.histogram_fixed_width(a, [0.0, 4], nbins=4, dtype=tf.float32)

# v = tf.get_variable('v', (), initializer=tf.constant_initializer(0.1))
# dist1 = ds.Normal(loc=0., scale=3.)
# dist2 = ds.Normal(loc=0., scale=v)
# # dist3 = ds.Mixture(cat=ds.Categorical(probs=[v, 1-v]), components=[dist1, dist2])
# loss = tf.distributions.kl_divergence(dist1, dist2)
# train = tf.train.AdamOptimizer(0.1).minimize(loss)


sess = tf.Session()
# sess.run(tf.global_variables_initializer())
# print(sess.run(v))
# sess.run(train)
# print(sess.run(v))

print(sess.run(loss))