import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.misc as misc
import random
from tensorflow.python.keras._impl.keras.datasets.cifar10 import load_data

####
#output
fout=open("output.txt","w")
#
learning_rate=0.001
training_epochs=2000
batch_size=100
fout.write("learning_rate : %d\n"%learning_rate)
fout.write("training_epochs : %d\n"%training_epochs)
fout.write("batch_size : %d\n"%batch_size)
####

label_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

(x_train, y_train), (x_test, y_test) = load_data()


X=tf.placeholder(tf.float32,[None,32,32,3])
Y=tf.placeholder(tf.int32,[None,1])
_Y=tf.reshape(tf.one_hot(Y,10),[-1,10])
training=tf.placeholder(tf.bool)

# convolutional layer #1
conv1=tf.layers.conv2d(inputs=X,filters=32,kernel_size=[3,3],padding="same",activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
pool1=tf.layers.max_pooling2d(inputs=conv1,pool_size=[2,2],padding="same",strides=2)
dropout1=tf.layers.dropout(inputs=pool1,rate=0.3,training=training)

# convolutional layer #2
conv2=tf.layers.conv2d(inputs=dropout1,filters=64,kernel_size=[3,3],padding="same",activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
pool2=tf.layers.max_pooling2d(inputs=conv2,pool_size=[2,2],padding="same",strides=2)
dropout2=tf.layers.dropout(inputs=pool2,rate=0.3,training=training)

# convolutional layer #3
conv3=tf.layers.conv2d(inputs=dropout2,filters=128,kernel_size=[3,3],padding="same",activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
pool3=tf.layers.max_pooling2d(inputs=conv3,pool_size=[2,2],padding="same",strides=2)
dropout3=tf.layers.dropout(inputs=pool3,rate=0.3,training=training)

# dense layer with relu
flat=tf.reshape(dropout3,[-1,4*4*128])
dense4=tf.layers.dense(inputs=flat,units=625,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
dropout4=tf.layers.dropout(inputs=dense4,rate=0.5,training=training)

# logit 
logits=tf.layers.dense(inputs=dropout4,units=10,kernel_initializer=tf.contrib.layers.xavier_initializer())

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=_Y))
optimizer=tf.train.AdamOptimizer(learning_rate).minimize(cost)

is_correct=tf.equal(tf.argmax(logits,1),tf.argmax(_Y,1))
accuracy=tf.reduce_mean(tf.cast(is_correct,"float"))





with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(training_epochs):
        avg_cost=0
        total_batch=int(len(x_train)/batch_size)
        for batch_x, batch_y in zip(np.split(x_train, total_batch), np.split(y_train, total_batch)):
            batch_cost,_=sess.run([cost,optimizer],feed_dict={X:batch_x,Y:batch_y,training:True})
            avg_cost+=batch_cost/total_batch
        fout.write("Epoch : %04d cost = %.9f\n"%(epoch+1,avg_cost))

    fout.write("Final Accuracy: %.9f\n"%sess.run(accuracy,feed_dict={X:x_test,Y:y_test,training:False}))

fout.close()
