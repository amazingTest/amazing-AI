import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.examples.tutorials import mnist

myMnist = mnist.input_data.read_data_sets("MNIST_data/", one_hot=True)
print(len(myMnist))
A = [[[1.0],[2.0],[3.0],[4.0],[5.0],[5.0]], [[11.0],[22.0],[33.0],[44.0],[55.0],[6.0]], [[113.0],[22.0],[33.0],[44.0],[55.0],[6.0]]]
B = [[3.0,2.0,3.0,4.0,5.0,6.0], [11.0,22.0,33.0,44.0,55.0,6.0], [113.0,22.0,33.0,44.0,55.0,6.0]]

with tf.Session() as sess:
    print(sess.run(tf.nn.softmax(A)))
    print(sess.run(tf.argmax(A, 2)))
    print(sess.run(tf.argmax(A, 0)))
    correct_prediction = sess.run(tf.equal(tf.argmax(A, 1), tf.argmax(B, 1)))
    print(correct_prediction)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), 1)
    print(sess.run(tf.cast(correct_prediction, "float")))
    print(sess.run(accuracy))