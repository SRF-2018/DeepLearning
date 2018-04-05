import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy import misc
from tensorflow.examples.tutorials.mnist import input_data

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

image_paths = ['TestImage/mnist_0.jpg', 'TestImage/mnist_1.jpg', 'TestImage/mnist_2.jpg', 'TestImage/mnist_3.jpg', 'TestImage/mnist_4.jpg', 'TestImage/mnist_5.jpg', 'TestImage/mnist_6.jpg', 'TestImage/mnist_7.jpg', 'TestImage/mnist_8.jpg', 'TestImage/mnist_9.jpg']

my_X = [];

for path in image_paths:
    img = misc.imread(path)
    gray = rgb2gray(img)
    gray = gray.reshape(-1)
    gray = [(255.0-x) / 255.0 for x in gray]
    my_X.append(gray)
nb_classes = 10

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])

W = tf.Variable(tf.random_normal([784, nb_classes]))
b = tf.Variable(tf.random_normal([nb_classes]))

hypothesis = tf.nn.softmax(tf.matmul(X, W)+b)

cost = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis), axis=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)

is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

training_epochs = 30
batch_size = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={X:batch_xs, Y:batch_ys})
            avg_cost += c / total_batch
        print('Epoch:', '%04d' % (epoch+1), 'cost = ', '{:.9f}'.format(avg_cost))
    print("Accuracy: ", accuracy.eval(session = sess, feed_dict={X:mnist.test.images, Y:mnist.test.labels}))
    print(sess.run(tf.arg_max(hypothesis, 1), feed_dict={X:my_X}))

