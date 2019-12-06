import tensorflow as tf

with tf.name_scope("t1"):
    x = tf.Variable([10], name='x')
    y = tf.Variable([20], name='y')
    z = tf.add(x, y, name='z')

with tf.name_scope("t2"):
    x = tf.Variable([10], name='x')
    y = tf.Variable([20], name='y')
    z = tf.add(x, y, name='z')

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    out = sess.run(z)
    print(z.name)

print(out)
