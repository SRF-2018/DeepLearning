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

r=1e-4

hypothesis=tf.nn.softmax(tf.matmul(X,W)+b)
cost=tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis),axis=1))+r*tf.reduce_sum(tf.square(W))
#hypothesis=tf.matmul(X,W)+b
#cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=hypothesis,labels=Y))
optimizer=tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)

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
    print("Accuracy: ",sess.run(accuracy,feed_dict={X:mnist.test.images,Y:mnist.test.labels}))
