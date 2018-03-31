import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.set_random_seed(777) 

num_points = 1000
vectors_set = []

for i in range(num_points):
    x = np.random.normal(0.0, 0.55)
    y = x*0.1+0.3 + np.random.normal(0.0, 0.03)
    vectors_set.append([x, y])

xy=np.array(vectors_set);

x_data=xy[:,0:-1]
y_data=xy[:,[-1]]

X = tf.placeholder(tf.float32, shape=[None, 1])
Y = tf.placeholder(tf.float32, shape=[None, 1])

W = tf.Variable(tf.random_normal([1, 1]),name='weight')
b = tf.Variable(tf.random_normal([1]),name='bias')


hypothesis = tf.matmul(X,W) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.005)
train = optimizer.minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(1000):
        cost_val, hy_val,_=sess.run([cost,hypothesis,train],feed_dict={X:x_data,Y:y_data})
        if step % 100 == 0:
            print(step, "Cost: ", cost_val, "\nPrediction:\n", hy_val)
            plt.plot(x_data, y_data, 'y*')  
            plt.plot(x_data, sess.run(W) * x_data + sess.run(b), 'r')
            plt.legend()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.show()
            plt.savefig("output/output%02d.png"%(step/100))
            plt.cla()

