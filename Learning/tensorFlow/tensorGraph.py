import tensorflow as tf

c = tf.constant(4.0)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
c_out = sess.run(c)
print(c_out)
print(c.graph == tf.get_default_graph())
print(c.graph)
print(tf.get_default_graph())