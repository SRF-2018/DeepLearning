import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

#size init
my_data_size = 50
my_test_size = 50

#0-'a', 1-'b', ...
ch = "abcde"
str = [chr(97 + x//10) for x in range(50)]

image_paths = ["mydata/%02d.png"%x for x in range(1,my_data_size + 1)]
test_paths = ["mydata/test%d.png"%x for x in range(1,my_test_size + 1)]
test_x = np.zeros((my_test_size, 784))
test_y = np.zeros((my_test_size, 1))

i = 0
my_X = np.zeros((my_data_size, 784))
my_Y = np.zeros((my_data_size, 1))

#training data preprocess
for path in image_paths:
    img = misc.imread(path)
    gray = rgb2gray(img)
    gray = gray.reshape(-1)
    gray = [(255.0-x) / 255.0 for x in gray]
    my_X[i] = gray
    my_Y[i][0] = i // 10
    i += 1

#test data preprocess
i = 0
for path in test_paths:
    img = misc.imread(path)
    gray = rgb2gray(img)
    gray = gray.reshape(-1)
    gray = [(255.0-x) / 255.0 for x in gray]
    test_x[i] = gray
    test_y[i] = i // 10
    i += 1

neural_num = 512
nb_classes = 5
keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.int32, [None, 1])
y_one_hot = tf.one_hot(y,nb_classes,dtype=tf.float32)
y_one_hot = tf.reshape(y_one_hot, [-1,nb_classes])

W1 = tf.get_variable("W1", shape = [784, neural_num],initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([neural_num]))
layer1 = tf.nn.relu(tf.matmul(x,W1) + b1)
layer1 = tf.nn.dropout(layer1, keep_prob=keep_prob)

W2 = tf.get_variable("W2", shape = [neural_num, neural_num],initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([neural_num]))
layer2 = tf.nn.relu(tf.matmul(layer1,W2) + b2)
layer2 = tf.nn.dropout(layer2, keep_prob=keep_prob)

W3 = tf.get_variable("W3", shape = [neural_num, nb_classes],initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([nb_classes]))
hypothesis = tf.matmul(layer2,W3) + b3

cost_i = tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis, labels=y_one_hot)
cost = tf.reduce_mean(cost_i)
optimizer= tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

predict = tf.argmax(hypothesis, 1)
curr_predict = tf.equal(predict, tf.argmax(y_one_hot, 1))
accuray = tf.reduce_mean(tf.cast(curr_predict, tf.float32))

file = open("nn_mnist_result.txt","w")
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(20000):
        sess.run(optimizer,feed_dict = {x : my_X, y : my_Y, keep_prob:0.7})

    pred = sess.run(predict, feed_dict = {x : test_x, keep_prob:1.0})
    tr = 0          #true predict
    fl = 0          #false predict
    for p, y in zip(pred, test_y.flatten()):
        file.write("step: {}, Real data: {}, Predict data: {}, result: {}\n".format(y, ch[int(y)], ch[p], int(y) == p))
        print("step: {}, Real data: {}, Predict data: {}, result: {}".format(y, ch[int(y)], ch[p], int(y) == p))
        tr += (p == int(y))
        fl += (p != int(y))
    file.write("accuracy : {:%}, test count : {}".format(float((tr)/ (tr+ fl)), tr + fl))
    print("accuracy : {:%}, test count : {}".format(float((tr)/ (tr+ fl)), tr + fl))