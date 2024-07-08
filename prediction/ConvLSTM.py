import tensorflow as tf
import numpy as np

LSTMStateTuple = tf.contrib.rnn.LSTMStateTuple
nest = tf.contrib.framework.nest


class ConvLSTMCell(tf.contrib.rnn.RNNCell):
    def __init__(self, shape, filter_size, num_features, forget_bias=1.0, initializer=None, use_peepholes=True,
                 state_is_tuple=True, activation=None, reuse=None):
        super(ConvLSTMCell, self).__init__(_reuse=reuse)
        self._shape = shape
        self._filter_size = filter_size
        self._num_features = num_features
        self._forget_bias = forget_bias
        self._state_is_tuple = state_is_tuple
        self._activation = activation or tf.tanh
        self._use_peepholes = use_peepholes
        self._initializer = initializer

    @property
    def state_size(self):
        return (LSTMStateTuple(self._num_features, self._num_features)
                if self._state_is_tuple else 2 * self._shape)

    @property
    def output_size(self):
        return self._num_features

    def call(self, inputs, state):
        sigmoid = tf.sigmoid
        dtype = inputs.dtype
        if self._state_is_tuple:
            c, h = state
        else:
            c, h = tf.split(value=state, num_or_size_splits=2, axis=3)

        scope = tf.get_variable_scope()
        with tf.variable_scope(scope, initializer=self._initializer) as scope:
            concat = _conv_linear([inputs, h], self._filter_size, 4 * self._num_features, True)

            i, j, f, o = tf.split(value=concat, num_or_size_splits=4, axis=3)

            if self._use_peepholes:
                w_f_diag = tf.get_variable("w_f_diag", shape=[self._num_features], dtype=dtype)
                w_i_diag = tf.get_variable("w_i_diag", shape=[self._num_features], dtype=dtype)
                w_o_diag = tf.get_variable("w_o_diag", shape=[self._num_features], dtype=dtype)

        if self._use_peepholes:
            new_c = (
                c * sigmoid(f + self._forget_bias + w_f_diag * c) + sigmoid(i + w_i_diag * c) * self._activation(j))
            new_h = self._activation(new_c) * sigmoid(o + w_o_diag * c)
        else:
            new_c = (c * sigmoid(f + self._forget_bias) + sigmoid(i) * self._activation(j))
            new_h = self._activation(new_c) * sigmoid(o)

        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_c, new_h)
        else:
            new_state = tf.concat([new_c, new_h], 3)
        return new_h, new_state


def _conv_linear(args, filter_size, num_features, bias, bias_initializer=None, bias_start=0.0):
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    total_arg_size_depth = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 4:
            raise ValueError("Linear is expecting 4D arguments: %s" % str(shapes))
        if not shape[3]:
            raise ValueError("Linear expects shape[4] of arguments: %s" % str(shapes))
        else:
            total_arg_size_depth += shape[3]

    dtype = [a.dtype for a in args][0]

    scope = tf.get_variable_scope()
    with tf.variable_scope(scope) as scope:
        matrix = tf.get_variable(
            "Matrix", [filter_size[0], filter_size[1], total_arg_size_depth, num_features], dtype=dtype)
        print(matrix)
        if len(args) == 1:
            res = tf.nn.conv2d(args[0], matrix, strides=[1, 1, 1, 1], padding='SAME')
        else:
            res = tf.nn.conv2d(tf.concat(axis=3, values=args), matrix, strides=[1, 1, 1, 1], padding='SAME')
        if not bias:
            return res
        if bias_initializer is None:
            bias_initializer = tf.constant_initializer(bias_start, dtype=dtype)
        bias_term = tf.get_variable("Bias", [num_features], dtype=dtype, initializer=bias_initializer)
        return res + bias_term
