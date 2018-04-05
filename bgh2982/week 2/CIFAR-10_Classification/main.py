import tensorflow as tf
import numpy as np
import random
import matplotlib.pyplot as plt
from tensorflow.python.keras._impl.keras.datasets.cifar10 import load_data

label_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

X = tf.placeholder(tf.float32, [None, 32, 32, 3])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

(x_train, y_train), (x_test, y_test) = load_data()
y_one_hot_train = tf.squeeze(tf.one_hot(y_train, 10), 1)
y_one_hot_test = tf.squeeze(tf.one_hot(y_test, 10), 1)

W1 = tf.Variable(tf.random_normal([5, 5, 3, 64], stddev=5e-2))
#b1 = tf.Variable(tf.random_normal([64]))
layer1 = tf.nn.relu(tf.nn.conv2d(X, W1, strides = [1, 1, 1, 1], padding='SAME'))
layer1 = tf.nn.max_pool(layer1, ksize=[1, 3, 3, 1], strides=[1,2,2,1], padding='SAME')
layer1 = tf.nn.dropout(layer1, keep_prob=keep_prob)

W2 = tf.Variable(tf.random_normal([5, 5, 64, 64], stddev=5e-2))
#b2 = tf.Variable(tf.random_normal([64]))
layer2 = tf.nn.relu(tf.nn.conv2d(layer1, W2, strides = [1, 1, 1, 1], padding='SAME'))
layer2 = tf.nn.max_pool(layer2, ksize=[1, 3, 3, 1], strides=[1,2,2,1], padding='SAME')
layer2 = tf.nn.dropout(layer2, keep_prob=keep_prob)

W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=5e-2))
#b3 = tf.Variable(tf.random_normal([128]))
layer3 = tf.nn.relu(tf.nn.conv2d(layer2, W3, strides = [1, 1, 1, 1], padding='SAME'))
layer3 = tf.nn.dropout(layer3, keep_prob=keep_prob)

W4 = tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=5e-2))
#b4 = tf.Variable(tf.random_normal([128]))
layer4 = tf.nn.relu(tf.nn.conv2d(layer3, W4, strides = [1, 1, 1, 1], padding='SAME'))
layer4 = tf.nn.dropout(layer4, keep_prob=keep_prob)

W5 = tf.Variable(tf.random_normal([3, 3, 128, 128], stddev=5e-2))
#b5 = tf.Variable(tf.random_normal([128]))
layer5 = tf.nn.relu(tf.nn.conv2d(layer4, W5, strides = [1, 1, 1, 1], padding='SAME'))
layer5 = tf.nn.dropout(layer5, keep_prob=keep_prob)
layer5 = tf.reshape(layer5, [-1, 8*8*128])

W6 = tf.Variable(tf.random_normal([8*8*128, 384], stddev=5e-2))
b6 = tf.Variable(tf.random_normal([384]))
layer6 = tf.nn.relu(tf.matmul(layer5, W6)+b6)
layer6 = tf.nn.dropout(layer6, keep_prob=keep_prob)

W7 = tf.get_variable("W5", shape=[384, 10], initializer = tf.contrib.layers.xavier_initializer())
b7 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(layer6, W7)+b7

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(1e-3).minimize(cost)

is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

training_epochs = 100
batch_size = 100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    y_one_hot_train = y_one_hot_train.eval()
    for epoch in range(training_epochs):
        avg_cost = 0;
        total_batch = int(len(x_train) / batch_size)
        for batch_x, batch_y in zip(np.split(x_train, batch_size), np.split(y_one_hot_train, batch_size)):
            c, _ = sess.run([cost, optimizer], feed_dict={X:batch_x, Y:batch_y, keep_prob:0.7})
            avg_cost += c/total_batch
        print('Epoch: ', '%04d' % (epoch+1), 'Cost = ', '{:.9f}'.format(avg_cost))
    total_batch = int(len(x_test) / batch_size)
    avg_accuracy = 0.;
    y_one_hot_test = y_one_hot_test.eval()
    for batch_x, batch_y in zip(np.split(x_test, batch_size), np.split(y_one_hot_test, batch_size)):
        a = sess.run(accuracy, feed_dict={X:batch_x, Y:batch_y, keep_prob:1})
        avg_accuracy += a/total_batch
    print("Accuracy: ", '{:.9f}'.format(avg_accuracy))
    for i in random.sample(range(1, int(len(x_test))), 10):
        plt.imshow(x_test[i])
        predict_label = sess.run(tf.arg_max(hypothesis, 1),
                                 feed_dict={X: np.expand_dims(x_test[i], 0), Y: np.expand_dims(y_one_hot_test[i], 0),
                                            keep_prob: 1})
        textstr = 'predicted : ' + label_name[int(predict_label)] + ", label    : " + label_name[y_test[i][0]]
        plt.text(1, 40, textstr, fontsize=14)
        plt.subplots_adjust(bottom=0.25)
        plt.show()
