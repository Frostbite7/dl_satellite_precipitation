import tensorflow as tf
from ConvLSTM import ConvLSTMCell
import numpy as np


def gen2_conv_lstm(shape1, shape2, filter_size1, filter_size2, num_features1, num_features2):
    lstm1 = ConvLSTMCell(shape=shape1, filter_size=filter_size1, num_features=num_features1)
    lstm2 = ConvLSTMCell(shape=shape2, filter_size=filter_size2, num_features=num_features2)
    multicell = tf.contrib.rnn.MultiRNNCell([lstm1, lstm2])
    return multicell


def gen_init_state(batch_size, shape1, shape2, num_features1, num_features2):
    init_state1 = tf.contrib.rnn.LSTMStateTuple(tf.zeros([batch_size, shape1[0], shape1[1], num_features1]),
                                                tf.zeros([batch_size, shape1[0], shape1[1], num_features1]))
    init_state2 = tf.contrib.rnn.LSTMStateTuple(tf.zeros([batch_size, shape2[0], shape2[1], num_features2]),
                                                tf.zeros([batch_size, shape2[0], shape2[1], num_features2]))
    init_state = (init_state1, init_state2)
    return init_state


def gen_init_state_np(batch_size, shape1, shape2, num_features1, num_features2):
    init_state_np = ((np.zeros([batch_size, shape1[0], shape1[1], num_features1]),
                      np.zeros([batch_size, shape1[0], shape1[1], num_features1])),
                     (np.zeros([batch_size, shape2[0], shape2[1], num_features2]),
                      np.zeros([batch_size, shape2[0], shape2[1], num_features2])))
    return init_state_np
