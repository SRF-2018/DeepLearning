import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

x_data = []
y_data = []
s = 0.0
for i in range(0, 100):
    x = s
    y = x + 3 + np.random.normal(0,0.05)
    x_data.append(x)
    y_data.append(y)
    s+=0.01

W = tf.Variable(tf.random_normal([1]),name='weight')
b = tf.Variable(tf.random_normal([1]),name='bias')
hypothesis = W*x_data + b
cost = tf.reduce_mean(tf.square(hypothesis - y_data))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train = optimizer.minimize(cost)

file = open('linear regression - result.txt','w')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(500):
        file.write('step : ' + str(step) + ', cost : ' + str(sess.run(cost)) + '\n')
        sess.run(train)
    file.write(str(sess.run(W)) + 'x + ' + str(sess.run(b)))
    plt.plot(x_data, y_data, 'y*')  
    plt.plot(x_data, sess.run(W) * x_data + sess.run(b), 'r')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
file.close()
