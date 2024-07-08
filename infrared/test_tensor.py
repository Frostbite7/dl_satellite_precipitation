import tensorflow as tf

pred = tf.placeholder(tf.bool)
# x = tf.Variable([1])
#
# def update_x_2():
#     assign_x_2 = tf.assign(x, [2])
#     with tf.control_dependencies([assign_x_2]):
#         return tf.identity(x)
#
# y = tf.cond(pred, update_x_2, lambda: tf.identity(x))
# with tf.Session() as sess:
#     sess.run(tf.global_variables_initializer())
#     print(sess.run(y, feed_dict={pred: True}))

if pred:
    a = tf.constant(0)
else:
    a = tf.constant(1)

with tf.Session() as sess:
    sess.run(a, feed_dict={pred: True})
    print(a)

# x = tf.get_variable('x', shape=[], initializer=tf.constant_initializer([0.0]))
# x_plus_1 = tf.assign_add(x, 1)
#
# with tf.control_dependencies([x_plus_1]):
#     z = tf.get_variable('d', [])
#     y = x
# init = tf.initialize_all_variables()
#
# with tf.Session() as session:
#     init.run()
#     for i in range(5):
#         print(z.eval())
#         print(y.eval())