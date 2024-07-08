import numpy as np
import os
import tensorflow as tf

dstpath = "upscaled/"
radarfile_path = 'radar/'
bghfile_path = 'goes/'
radar_pre = 'q2hrus'
bgh_pre = 'bghrus'
batch_size = 50
batch_number = 10


def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f

bgh = tf.placeholder(tf.float32, shape=[None, 784])
rain = tf.placeholder(tf.float32, shape=[None, 2])

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')


bgh_image = tf.reshape(bgh, [-1, 28, 28, 1])
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(bgh_image, W_conv1) + b_conv1)
h_conv1 = max_pool_(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
h_conv2 = max_pool_(h_conv2)

W_fc1 = weight_variable([7 * 7 * 64, 256])
b_fc1 = bias_variable([256])
h_conv2_flat = tf.reshape(h_conv2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([256, 2])
b_fc2 = bias_variable([2])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=rain, logits=y_conv))
train_step = tf.train.AdamOptimizer(0.1).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(rain, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

bgh_list = list(listdir_nohidden('upscaled/goes'))
rain_list = list(listdir_nohidden('upscaled/radar'))
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(batch_number):
        bgh_batch = np.zeros((batch_size, 784))
        rain_batch = np.zeros((batch_size, 2))
        for j in range(batch_size):
            bgh_batch[j, ] = np.fromfile('upscaled/goes/' + bgh_list[i * batch_size + j])
            rain_batch[j, ] = np.fromfile('upscaled/radar/' + rain_list[i * batch_size + j])
        if i % 1 == 0:
            train_accuracy = accuracy.eval(feed_dict={
                bgh: bgh_batch, rain: rain_batch, keep_prob: 1.0})
            print('step %d, training accuracy %g' % (i, train_accuracy))
        train_step.run(feed_dict={bgh: bgh_batch, rain: rain_batch, keep_prob: 0.5})
