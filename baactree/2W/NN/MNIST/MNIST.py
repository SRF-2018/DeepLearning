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
classes=10

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
Y=tf.placeholder("float",[None,classes])

lc=30
lr=1
print("lc : ",lc,"lr : ",lr)
W1=tf.Variable(tf.random_normal([784,lc]))
b1=tf.Variable(tf.random_normal([1,lc]))
layer=tf.nn.softmax(tf.matmul(X,W1)+b1)

W2=tf.Variable(tf.random_normal([lc,classes]))
b2=tf.Variable(tf.random_normal([1,classes]))

hypothesis=tf.nn.softmax(tf.matmul(layer,W2)+b2)

r=1e-3
cost=tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypothesis),1))+r*tf.reduce_sum(tf.square(W1))+r*tf.reduce_sum(tf.square(W2))
optimizer=tf.train.GradientDescentOptimizer(lr).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(20001):
        nhy,ncost,_=sess.run([hypothesis,cost,optimizer],feed_dict={X:x_data,Y:y_data})
        if step%1000==0:
            print("step : ",step, "cost : ", sess.run(tf.reduce_mean(ncost)))
    ac=0
    for check in range(20):
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
        print(sess.run(hypothesis,feed_dict={X:x_test}))
        labels=sess.run(tf.argmax(y_test,1))
        prediction=sess.run(tf.argmax(hypothesis,1),feed_dict={X:x_test})
        print("Label: ",labels)
        print("Prediction: ",prediction)
        if labels==prediction:
            ac+=1
    print("accuracy %d/%d = %lf"%(ac,20,ac/20))
