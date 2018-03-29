import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py

import tensorflow as tf
import matplotlib.pyplot as plt
import random
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
from sklearn.preprocessing import MinMaxScaler
mnist=input_data.read_data_sets("MNIST_data/",one_hot=True)


nb_classes=10
X=tf.placeholder("float",[None,784])
Y=tf.placeholder("float",[None,nb_classes])

W=tf.Variable(tf.random_normal([784,nb_classes]))
b=tf.Variable(tf.random_normal([nb_classes]))

hypothesis=tf.nn.softmax(tf.matmul(X,W)+b)

cost=tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis),axis=1))
optimizer=tf.train.GradientDescentOptimizer(learning_rate=1e-2).minimize(cost)

is_correct=tf.equal(tf.argmax(hypothesis,1),tf.argmax(Y,1))
accuracy=tf.reduce_mean(tf.cast(is_correct,"float"))

# count of training all dataset 
training_epochs=15
batch_size=100

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost=0
        # 올림 아닌가
        total_batch=int(mnist.train.num_examples/batch_size)

        for i in range(total_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            #scaler=MinMaxScaler(feature_range=(0,1))
            #batch_xs=scaler.fit_transform(batch_xs)
            #batch_xs=MinMaxScaler(batch_xs)
        
            c,_=sess.run([cost,optimizer],feed_dict={X:batch_xs,Y:batch_ys})
            avg_cost+=c/total_batch
        print("Epoch :","%04d"%(epoch+1),"cost =","{:.9f}".format(avg_cost))
    r=random.randint(0,mnist.test.num_examples-1)
    print("Label: ",sess.run(tf.argmax(mnist.test.labels[r:r+1],1)))
    print("Prediction: ",sess.run(tf.argmax(hypothesis,1),feed_dict={X:mnist.test.images[r:r+1]}))
    plt.imshow(mnist.test.images[r:r+1].reshape(28,28),cmap='Greys',interpolation='nearest')
    plt.show()
