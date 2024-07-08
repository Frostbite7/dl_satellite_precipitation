import numpy as np
import os
import tensorflow as tf

from random import shuffle
from post_proce import results

train_path = 'train/'
rain_path = '1/'
norain_path = '0/'
rain_pre = '1_'
norain_pre = '0_'
val_path = 'val/'
test_path = 'test/'

rain_number = 400
norain_number = 400
rain_multiplier = 1
norain_multiplier = 1
n_epoch = 20
batch_size = 100
batch_number = int((rain_number * rain_multiplier + norain_number * norain_multiplier) / batch_size)

n_test = 50
n_val = 50
val_batch_size = 10
val_batch_number = int(n_val / val_batch_size)
test_batch_number = int(n_test / val_batch_size)

bn_epsilon = 0.001
height = 29
width = 29


def listdir_nohidden(path):
    for f in os.listdir(path):
        if not f.startswith('.'):
            yield f


def weight_variable(name, shape):
    return tf.get_variable(name=name, shape=shape, initializer=tf.truncated_normal_initializer(stddev=0.01))


def bias_variable(name, shape):
    return tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(0.1))


def scale_variable(name, shape):
    return tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(1))


def beta_variable(name, shape):
    return tf.get_variable(name=name, shape=shape, initializer=tf.constant_initializer(0))


def bn(x, name, is_training, decay=0.99):
    with tf.variable_scope(name):
        x_shape = x.get_shape()
        params_shape = x_shape[-1]
        axis = list(range(len(x_shape) - 1))

        beta = beta_variable('beta', params_shape)
        # scale = scale_variable('scale', params_shape)
        batch_mean, batch_var = tf.nn.moments(x, axis, name='moments')
        ema = tf.train.ExponentialMovingAverage(decay=decay, zero_debias=True)

        def mean_var_with_update():
            ema_apply_op = ema.apply([batch_mean, batch_var])
            with tf.control_dependencies([ema_apply_op]):
                return tf.identity(batch_mean), tf.identity(batch_var)

        mean, var = tf.cond(is_training,
                            mean_var_with_update,
                            lambda: (ema.average(batch_mean), ema.average(batch_var)))
        normed = tf.nn.batch_normalization(x, mean, var, beta, None, bn_epsilon)
    return normed


def cnn(x, is_training, keep_prob):
    with tf.variable_scope('reshape'):
        x_image = tf.reshape(x, [-1, height, width, 1])

    with tf.variable_scope('conv1'):
        W_conv1 = weight_variable('w', [5, 5, 1, 8])
        h_conv1 = tf.nn.conv2d(x_image, W_conv1, strides=[1, 1, 1, 1], padding='VALID')
        h_conv1_bn = tf.layers.batch_normalization(h_conv1, momentum=0.99, scale=False, training=is_training)
        # h_conv1_bn = bn(h_conv1, 'bn', is_training)
        h_conv1_fl = tf.nn.relu(h_conv1_bn)

    with tf.variable_scope('conv2'):
        W_conv2 = weight_variable('w', [5, 5, 8, 8])
        h_conv2 = tf.nn.conv2d(h_conv1_fl, W_conv2, strides=[1, 1, 1, 1], padding='VALID')
        h_conv2_bn = tf.layers.batch_normalization(h_conv2, momentum=0.99, scale=False, training=is_training)
        # h_conv2_bn = bn(h_conv2, 'bn', is_training)
        h_conv2_fl = tf.nn.relu(h_conv2_bn)

    with tf.variable_scope('fc1'):
        h_conv2_flat = tf.reshape(h_conv2_fl, [-1, (height - 8) * (width - 8) * 8])
        W_fc1 = weight_variable('W', [(height - 8) * (width - 8) * 8, 16])
        b_fc1 = bias_variable('b', [16])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

    with tf.variable_scope('dropout'):
        h_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.variable_scope('fc2'):
        W_fc2 = weight_variable('W', [16,1])
        b_fc2 = bias_variable('b', [1])
        y_cnn = tf.matmul(h_drop, W_fc2) + b_fc2

    return tf.reshape(y_cnn, [-1])


def main():
    x = tf.placeholder(tf.float32, [None, height, width, 1])
    y = tf.placeholder(tf.float32, [None])
    phase_train = tf.placeholder(tf.bool)
    keep_prob = tf.placeholder(tf.float32)

    with tf.variable_scope('cnn1'):
        y_conv = cnn(x, phase_train, keep_prob)

    with tf.variable_scope('loss'):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_conv)
        loss = tf.reduce_mean(cross_entropy)

    with tf.variable_scope('Adam_optimizer'):
        train_step = tf.train.AdamOptimizer(1e-3).minimize(loss)

    # with tf.variable_scope('accuracy'):
    #     correct_prediction = tf.equal(tf.arg_max(y_conv, 1), tf.arg_max(y, 1))
    #     correct_prediction = tf.cast(correct_prediction, tf.float32)
    # accuracy = tf.reduce_mean(correct_prediction)

    rain_number_list = [i + 1 for i in range(rain_number)]
    norain_number_list = [i + 1 for i in range(norain_number)]
    rain_list = [rain_path + rain_pre + '{}.bin'.format(element) for element in
                 rain_number_list] * rain_multiplier
    norain_list = [norain_path + norain_pre + '{}.bin'.format(element) for element in
                   norain_number_list] * norain_multiplier
    sample_list = rain_list + norain_list
    shuffle(sample_list)
    sample_list = sample_list * n_epoch

    val_list = list(listdir_nohidden(val_path))
    test_list = list(listdir_nohidden(test_path))

    with tf.Session() as sess:
        cnt_noimprove = 0
        min_loss = 10 ** 10
        early_stop_time = 100

        sess.run(tf.global_variables_initializer())
        for i in range(batch_number * n_epoch):
            bgh_batch = np.zeros([batch_size, height, width, 1])
            rain_batch = np.zeros([batch_size])
            for j in range(batch_size):
                sample = np.reshape(np.fromfile(train_path + sample_list[i * batch_size + j]), [height, width, 2])
                bgh_batch[j,] = sample[:, :, 0:1]
                rain_batch[j] = 1 if sample[14, 14, 1] >= 0.1 else 0

            if i % 1 == 0:
                # train_loss, y_predict= sess.run([loss, y_conv],
                #                                  feed_dict={x: bgh_batch, y: rain_batch, phase_train: True,
                #                                             keep_prob: 1.0})
                # print('identity, step %d, training loss %g' % (i, train_loss))
                # csi, pod, far = results(y_predict, rain_batch)
                # print('identity, step %d, training csi: %g, pod: %g, far: %g' % (i, csi, pod, far))
                train_loss, y_predict = sess.run([loss, y_conv],
                                                 feed_dict={x: bgh_batch, y: rain_batch, phase_train: False,
                                                            keep_prob: 1.0})
                print('ema, step %d, training loss %g' % (i, train_loss))
                csi, pod, far = results(y_predict, rain_batch)
                print('ema, step %d, training csi: %g, pod: %g, far: %g' % (i, csi, pod, far))
            train_step.run(feed_dict={x: bgh_batch, y: rain_batch, phase_train: True, keep_prob: 0.5})

            if i % 2 == 0:
                val_loss = []
                csi = []
                pod = []
                far = []
                for k in range(val_batch_number):
                    bgh_batch_val = np.zeros([val_batch_size, height, width, 1])
                    rain_batch_val = np.zeros([val_batch_size])
                    for j in range(val_batch_size):
                        sample = np.reshape(np.fromfile(val_path + val_list[k * val_batch_size + j]),
                                                [height, width, 2])
                        bgh_batch_val[j,] = sample[:, :, 0:1]
                        rain_batch_val[j] = 1 if sample[14, 14, 1] >= 0.1 else 0

                    val_loss_, y_predict = sess.run([loss, y_conv],
                                                    feed_dict={x: bgh_batch_val, y: rain_batch_val, phase_train: False,
                                                                keep_prob: 1.0})
                    val_loss.append(val_loss_)
                    csi_, pod_, far_ = results(y_predict, rain_batch_val)
                    csi.append(csi_)
                    pod.append(pod_)
                    far.append(far_)

                val_loss_mean = np.mean(val_loss)
                print('step %d, validation loss %g' % (i, val_loss_mean))
                print('step %d, validation csi: %g, pod: %g, far: %g' % (i, np.mean(csi), np.mean(pod), np.mean(far)))

                if val_loss_mean < min_loss:
                    min_loss = val_loss_mean
                    cnt_noimprove = 0
                else:
                    cnt_noimprove = cnt_noimprove + 1
                if cnt_noimprove > early_stop_time:
                    print('step %d, early stop testing loss %g' % (i, val_loss_mean))
                    break

                # test_loss = []
                # csi = []
                # pod = []
                # far = []
                # for k in range(test_batch_number):
                #     bgh_batch_test = np.zeros([val_batch_size, height, width, 1])
                #     rain_batch_test = np.zeros([val_batch_size])
                #     for j in range(val_batch_size):
                #         sample = np.reshape(np.fromfile(test_path + test_list[k * val_batch_size + j]),
                #                             [height, width, 2])
                #         bgh_batch_test[j,] = sample[:, :, 0:1]
                #         rain_batch_test[j] = 1 if sample[14, 14, 1] >= 0.1 else 0
                #
                #     test_loss_, y_predict = sess.run([loss, y_conv],
                #                                      feed_dict={x: bgh_batch_test, y: rain_batch_test, phase_train: False,
                #                                                 keep_prob: 1.0})
                #     test_loss.append(test_loss_)
                #     csi_, pod_, far_ = results(y_predict, rain_batch_test)
                #     csi.append(csi_)
                #     pod.append(pod_)
                #     far.append(far_)
                #
                # test_loss_mean = np.mean(test_loss)
                # print('step %d, test loss %g' % (i, test_loss_mean))
                # print('step %d, test csi: %g, pod: %g, far: %g' % (i, np.mean(csi), np.mean(pod), np.mean(far)))


if __name__ == '__main__':
    main()
