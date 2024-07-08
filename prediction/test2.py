import tensorflow as tf

logs_path = 'test_logs/'

def func(x):
    state = tf.get_variable('name', [1], tf.float32, tf.constant_initializer(5))
    one = tf.constant(1, dtype=tf.float32)
    out = x * state
    return out

a = tf.placeholder(dtype=tf.float32)
with tf.variable_scope('1') as scope:
    print(scope.reuse)
    b = func(a)
    tf.get_variable_scope().reuse_variables()
    c = func(b)

print(a.shape)
print(c.shape)

tf.summary.scalar('c_result', a)
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter(logs_path)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    summary, result = sess.run([merged, c], feed_dict={a: 4})
    writer.add_summary(summary, 1)
    print(result)
