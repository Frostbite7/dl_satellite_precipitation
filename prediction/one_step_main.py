import tensorflow as tf
import numpy as np
# import matplotlib.pyplot as plt

from post_processing import results
from layer_def import gen2_conv_lstm
from layer_def import gen_init_state
from layer_def import gen_init_state_np

path = 'rain/'
path_test = 'rain_test/'
log_path = 'onestep_log/'
batch_size = 2
num_epoch = 100
num_step = 10
epoch_size = 1

num_distinct_chan = 2
distinct_chan = [1, 1]

shape1 = [(10, 10), (10, 10)]
filter_size1 = [(3, 3), (3, 3)]
num_features1 = [4, 4]
shape2 = [(10, 10), (10, 10)]
filter_size2 = [(3, 3), (3, 3)]
num_features2 = [4] * num_distinct_chan
filter_size3 = (1, 1)
num_features3 = 1
b_start = 0.0

cells = []
init_states = []
init_states_np = []
for i in range(num_distinct_chan):
    cells.append(
        gen2_conv_lstm(shape1[i], shape2[i], filter_size1[i], filter_size2[i], num_features1[i], num_features2[i]))
    init_states.append(gen_init_state(batch_size, shape1[i], shape2[i], num_features1[i], num_features2[i]))
    init_states_np.append(gen_init_state_np(batch_size, shape1[i], shape2[i], num_features1[i], num_features2[i]))
init_states = tuple(init_states)
init_states_np = tuple(init_states_np)

x_placeholder = tf.placeholder(tf.float32, [batch_size, num_step, shape1[0][0], shape1[0][1], 2])
y_placeholder = tf.placeholder(tf.float32, [batch_size, num_step, shape1[0][0], shape1[0][1], 1])
inputs_list = tf.unstack(x_placeholder, axis=1)
one = tf.constant(1.0)

states = list(init_states)
h2s = [0] * num_distinct_chan
outputs_list = []
states_list = []
with tf.variable_scope('one_step_cell') as scope:
    for i, inputs in enumerate(inputs_list):
        if i > 0:
            scope.reuse_variables()
        inputs = tf.split(inputs, distinct_chan, axis=3)
        # inputs = [inputs]
        # h2s[0], states = multicell(inputs, states)
        for j in range(num_distinct_chan):
            with tf.variable_scope('dist_channel{}'.format(j)):
                h2s[j], states[j] = cells[j](inputs[j], states[j])
        outputs_list.append(h2s[:])
        states_list.append(states[:])
final_h2 = outputs_list[-1]
final_state = states_list[-1]

with tf.variable_scope('predict'):
    w_h = tf.get_variable('weights', [filter_size3[0], filter_size3[1], sum(num_features2), num_features3], tf.float32)
    b_h = tf.get_variable('bias', [num_features3], tf.float32, tf.constant_initializer(b_start))
y_list_ = tf.unstack(y_placeholder, axis=1)
predictions_ = [tf.nn.conv2d(tf.concat(output_, axis=3), w_h, strides=[1, 1, 1, 1], padding='SAME') + b_h for output_ in
                outputs_list]
y_list = y_list_[1:]
predictions = predictions_[:-1]

losses = [-(y_ * tf.log(predict) + (one - y_) * tf.log(one - predict)) for predict, y_ in zip(predictions, y_list)]
# losses = [(predict - y_) * (predict - y_) for predict, y_ in zip(predictions, y_list)]
loss1 = [tf.reshape(loss, [-1]) for loss in losses]
loss2 = [tf.reduce_mean(loss, 0) for loss in loss1]
total_loss = tf.reduce_mean(loss2, 0)
train_step = tf.train.AdamOptimizer(0.01).minimize(total_loss)

tf.summary.scalar('loss', total_loss)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(log_path, graph=tf.get_default_graph())

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    training_losses = []
    for i in range(num_epoch):
        X = np.fromfile(path + str(i) + '.bin')
        X = np.reshape(X, [batch_size, num_step * epoch_size, shape1[0][0], shape1[0][1], 2])
        train_state = init_states_np
        for j in range(epoch_size):
            X_in = X[:, j * num_step:(j + 1) * num_step, :, :, :]
            Y_in = X[:, j * num_step:(j + 1) * num_step, :, :, 0:1]
            output_r, summary, train_loss, train_state, _ = sess.run(
                [outputs_list, merged, total_loss, final_state, train_step],
                feed_dict={x_placeholder: X_in, y_placeholder: Y_in,
                           init_states: train_state})
            training_losses.append(train_loss)
            print('Loss:', train_loss)
            # output_r = np.array(output_r)
            # print('output:', output_r[:, 0, 0, 0, 0, 0])
            writer.add_summary(summary, i * epoch_size + j)

            # with tf.Session() as sess:
    Correlation = []
    CSI = []
    FAR = []
    POD = []
    for i in range(10):
        X_t = np.fromfile(path_test + str(i) + '.bin')
        X_t = np.reshape(X_t, [batch_size, num_step * epoch_size, shape1[0][0], shape1[0][1], 2])
        train_state = init_states_np
        correlation = []
        csi = []
        far = []
        pod = []
        for j in range(epoch_size):
            X_in = X_t[:, j * num_step:(j + 1) * num_step, :, :, :]
            Y_in = X_t[:, j * num_step:(j + 1) * num_step, :, :, 0:1]
            train_state, predict = sess.run([final_state, predictions],
                                            feed_dict={x_placeholder: X_in, init_states: train_state})

            correlation_, csi_, far_, pod_ = results(predict, Y_in[:, 1:, :, :, :])
            correlation.extend(correlation_)
            csi.extend(csi_)
            far.extend(far_)
            pod.extend(pod_)

        Correlation.append(correlation)
        CSI.append(csi)
        FAR.append(far)
        POD.append(pod)

Correlation = np.mean(Correlation, 0)
CSI = np.mean(CSI, 0)
FAR = np.mean(FAR, 0)
POD = np.mean(POD, 0)

print(Correlation, CSI, FAR, POD)
