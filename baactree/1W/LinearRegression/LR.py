import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# make training data

num_points=1000
vectors_set=[]

for i in range(num_points):
    x1=np.random.normal(0.0,0.55)
    y1=x1*0.1+0.3+np.random.normal(0.0,0.03)
    vectors_set.append([x1,y1])

xy=np.array(vectors_set)

x_data=xy[:,0:-1]
y_data=xy[:,[-1]]

X=tf.placeholder("float",shape=[None,1])
Y=tf.placeholder("float",shape=[None,1])

W=tf.Variable(tf.random_normal([1,1]),name='weight')
b=tf.Variable(tf.random_normal([1]),name='bias')

hypothesis=tf.matmul(X,W)+b

cost=tf.reduce_mean(tf.square(hypothesis-Y))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(1000):
        cost_val,hy_val,_=sess.run([cost,hypothesis,optimizer],feed_dict={X:x_data,Y:y_data})
        if step%50==0:
            print("step : ",step,"cost mean : ",sess.run(tf.reduce_mean(cost_val)))
            plt.plot(x_data,y_data,'ro',label="r")
            plt.plot(x_data,sess.run(W)*x_data+sess.run(b),label="b")
            plt.legend()
            plt.xlabel('x')
            plt.ylabel('y')
            plt.savefig("output/output%02d.png"%(step/50))
            plt.cla()




