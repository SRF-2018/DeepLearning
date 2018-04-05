import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

num_points = 1000
vectors_set = []

for i in range(num_points):
    x1 = np.random.normal(0.0, 0.55)
    y1 = x1*0.1+0.3 + np.random.normal(0.0, 0.03)
    vectors_set.append([x1, y1])

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

plt.plot(x_data, y_data, 'ro')
plt.legend()
plt.xlim([-2, 2])
plt.ylim([0.0, 0.5])
plt.show()

W = tf.Variable(tf.random_normal([1]), name = 'weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = W*x_data + b;

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

optimizer = tf.train.GradientDescentOptimizer(learning_rate = 0.5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
for step in range(15):
    sess.run(train)
    print(step, "\nW: ", sess.run(W), "\nb: ", sess.run(b))
    plt.plot(x_data, y_data, 'ro')
    plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim([-2, 2])
    plt.ylim([0.0, 0.5])
    fig = plt.gcf() #변경한 곳
    plt.show()
    fig.savefig('linearRegression'+str(step+1)+'.png') #변경한 곳