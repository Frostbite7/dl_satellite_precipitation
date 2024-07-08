import tensorflow as tf

a = tf.placeholder(tf.float32, [1])
is_training = tf.placeholder(tf.bool)
mean, var = tf.nn.moments(a, [0])
ema = tf.train.ExponentialMovingAverage(decay=0.5, zero_debias=True)

def mean_var_with_update():
    ema_apply_op = ema.apply([mean])
    with tf.control_dependencies([ema_apply_op]):
        return tf.identity(mean)

mean_ema = tf.cond(is_training, mean_var_with_update, lambda: ema.average(mean))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10, 200, 10):
        # ema_apply_op.run(feed_dict={a: [i+0.1]})
        c = sess.run(mean_ema, feed_dict={a: [i], is_training: False})
        d = sess.run(mean_ema, feed_dict={a: [i], is_training: False})
        # e = sess.run(mean_ema, feed_dict={a: [i + 0.1], is_training: True})
        # f = sess.run(mean_ema, feed_dict={a: [i + 0.1], is_training: False})
        print(c, d)
