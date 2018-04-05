import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from scipy import misc

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

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)


W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=5e-2))
L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=5e-2))
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=5e-2))
L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[1,2,2,1], padding='SAME')
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
L3 = tf.reshape(L3, [-1, 128*4*4])

W4 = tf.get_variable("W4", shape=[128*4*4, 625], initializer = tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([625]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob = keep_prob)

W5 = tf.get_variable("W5", shape=[625, 10], initializer = tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L4, W5) + b5

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

training_epochs = 15
batch_size = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={X:batch_x, Y:batch_y, keep_prob:0.7})
            avg_cost += c / total_batch
        print('Epoch: ', '%04d' % (epoch+1), 'Cost = ', '{:.9f}'.format(avg_cost))
    total_batch = int(mnist.test.num_examples / batch_size)
    avg_accuracy = 0.;
    for i in range(batch_size):
        batch_x, batch_y = mnist.test.next_batch(batch_size)
        a = sess.run(accuracy, feed_dict={X:batch_x, Y:batch_y, keep_prob:1})
        avg_accuracy += a/total_batch
    print("Accuracy: ", '{:.9f}'.format(avg_accuracy))
    print(sess.run(tf.arg_max(hypothesis, 1), feed_dict={X:my_X, keep_prob:1}))