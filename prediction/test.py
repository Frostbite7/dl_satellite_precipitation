import tensorflow as tf
import numpy as np

a = tf.placeholder(tf.float32, [2, 4, 4, None])
kernel = tf.ones([2, 2, 5, 3])
b = tf.zeros([4, 4,1])
a_ = np.ones([2, 4, 4, 5])

with tf.device('/cpu:0'):
    c = tf.nn.conv2d(a, kernel, [1, 1, 1, 1], padding='SAME')

d = c + b

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    e = sess.run(a, feed_dict={a: a_})
    print(e)
    print(type(e))
