import tensorflow as tf
import numpy as np
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

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])

keep_prob = tf.placeholder(tf.float32)

W1 = tf.get_variable('weight1', shape=[784, 512], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([512]), name='bias1')
with tf.name_scope("layer1") as scope:
    layer1 = tf.nn.relu(tf.matmul(tf.layers.batch_normalization(X), W1) + b1)
    layer1 = tf.nn.dropout(layer1, keep_prob=keep_prob)

W2 = tf.get_variable('weight2', shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([512]), name='bias2')
with tf.name_scope("layer2") as scope:
    layer2 = tf.nn.relu(tf.matmul(tf.layers.batch_normalization(layer1), W2) + b2)
    layer2 = tf.nn.dropout(layer2, keep_prob=keep_prob)

W3 = tf.get_variable('weight3', shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([512]), name='bias3')
with tf.name_scope("layer3") as scope:
    layer3 = tf.nn.relu(tf.matmul(tf.layers.batch_normalization(layer2), W3) + b3)
    layer3 = tf.nn.dropout(layer3, keep_prob=keep_prob)

W4 = tf.get_variable('weight4', shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([512]), name='bias3')
with tf.name_scope("layer4") as scope:
    layer4 = tf.nn.relu(tf.matmul(tf.layers.batch_normalization(layer3), W4) + b4)
    layer4 = tf.nn.dropout(layer4, keep_prob=keep_prob)

W5 = tf.get_variable('weight5', shape=[512, 10], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]), name='bias3')
with tf.name_scope("layer5") as scope:
    hypothesis = tf.matmul(tf.layers.batch_normalization(layer4), W5) + b5

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

training_epochs = 15
batch_size = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost = 0
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(batch_size):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            c, _ = sess.run([cost, optimizer], feed_dict={X:batch_x, Y:batch_y, keep_prob:0.7})
            avg_cost += c / total_batch
        print('Epoch: ', '%04d' % (epoch+1), 'Cost = ', '{:.9f}'.format(avg_cost))
    print("Accuracy: ", accuracy.eval(session = sess, feed_dict={X:mnist.test.images, Y:mnist.test.labels, keep_prob:1}))
    print(sess.run(tf.arg_max(hypothesis, 1), feed_dict={X:my_X, keep_prob:1}))