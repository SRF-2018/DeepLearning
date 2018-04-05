import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
#xy = np.loadtxt('data-03-diabetes.csv', delimiter=',', dtype=np.float32)

#x_data = xy[:, 0:-1]
#y_data = xy[:, [-1]]

num_points = 2000
x_data = []
y_data = []
for i in range(num_points):
    if np.random.random() > 0.5:
        x_data.append([np.random.normal(0.0, 0.9), np.random.normal(0.0, 0.9)])
        y_data.append([0])
    else:
        x_data.append([np.random.normal(3.0, 0.5), np.random.normal(1.0, 0.5)])
        y_data.append([1])

df = pd.DataFrame({"x":[v[0] for v in x_data], "y":[v[1] for v in x_data]})
sns.lmplot("x", "y", data=df, fit_reg=False, size=6)
plt.show()

print(x_data)

X = tf.placeholder(tf.float32, shape = [None, 2])
Y = tf.placeholder(tf.float32, shape = [None, 1])

W = tf.Variable(tf.random_normal([2, 1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')

hypothesis = tf.sigmoid(tf.matmul(X, W)+b)

cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))

train = tf.train.GradientDescentOptimizer(learning_rate=1e-2).minimize(cost)

predicted = tf.cast(hypothesis>0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(4401):
        cost_val, _ = sess.run([cost, train], feed_dict={X:x_data, Y:y_data})
        if step %400 == 0:
            h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X: x_data, Y: y_data})
            print(step, cost_val)
            data = {"x": [], "y": [], "cluster": []}
            c = c.reshape(-1)
            print(c)
            for i in range(len(c)):
                data["x"].append(x_data[i][0])
                data["y"].append(x_data[i][1])
                if c[i] > 0.5:
                    data["cluster"].append(1)
                else:
                    data["cluster"].append(0)
            print(data)
            df = pd.DataFrame(data)

            sns.lmplot("x", "y", data=df, fit_reg=False, size=6, hue="cluster", legend=False)
            plt.show()

    h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})
    print("\nHypothesis : \n", h, "\nCorrect (Y) : \n", c, "\nAccuracy: ", a)

