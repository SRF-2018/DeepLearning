import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.misc as misc
def rgb2gray(rgb):
    return np.dot(rgb[:,:,:3],[0.299,0.587,0.114])

image_paths=["TrainingImage/%02d.png"%i for i in range(100)]

x_data=[]
y_data=[]

for path in image_paths:
    img=misc.imread(path)
    gray=rgb2gray(img)
    gray=gray.reshape(-1)
    gray=[(255.0-x)/255.0 for x in gray]
    x_data.append(gray)
    y=[ 0 for i in range(10)]
    y[len(y_data)//10]=1
    y_data.append(y)


X=tf.placeholder("float",[None,784])
X_img=tf.reshape(X,[-1,28,28,1])
Y=tf.placeholder("float",[None,10])
training=tf.placeholder("bool")

# convolutional layer #1
conv1=tf.layers.conv2d(inputs=X_img,filters=32,kernel_size=[3,3],padding="same",activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
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

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=Y))
optimizer=tf.train.AdamOptimizer(0.001).minimize(cost)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(5001):
        ncost,_=sess.run([cost,optimizer],feed_dict={X:x_data,Y:y_data,training:True})
        if step%1000==0:
            print("step : ",step, "cost : ", sess.run(tf.reduce_mean(ncost)))
    ac=0
    for check in range(100):
        r=np.random.randint(0,9)*10+np.random.randint(0,1)
        npath="TestImage/%02d.png"%r
        x_test=[]
        y_test=[]
        img=misc.imread(npath)
        gray=rgb2gray(img)
        gray=gray.reshape(-1)
        gray=[(255.0-x)/255.0 for x in gray]
        x_test.append(gray)
        y=[0 for i in range(10)]
        y[r//10]=1
        y_test.append(y)
        print(y_test)
        print(sess.run(logits,feed_dict={X:x_test,training:False}))
        labels=sess.run(tf.argmax(y_test,1))
        prediction=sess.run(tf.argmax(logits,1),feed_dict={X:x_test,training:False})
        print("Label: ",labels)
        print("Prediction: ",prediction)
        if labels==prediction:
            ac+=1
    print("accuracy %d/%d = %lf"%(ac,100,ac/100))
