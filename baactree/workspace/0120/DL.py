import tensorflow as tf
import numpy as np
import random
from main import alpha

def make_data(num):
    xd=[]
    yd=[]
    for size in range(num):
        xi=np.zeros((128))
        type=random.randrange(0,10)
        y=random.randrange(0,9)
        yi=np.zeros((10))
        yi[type]=1
        for i in range(8):
            for j in range(8):
                xi[(i+y)*8+j]=alpha[type][i][j]
        for i in range(16):
            for j in range(8):
                if random.randrange(0,10)==0:
                    xi[i*8+j]=(8 if xi[i*8+j]==0 else 0)
                xi[i*8+j]=(1 if xi[i*8+j]==8 else 0)
       
        xd.append(xi)
        yd.append(yi)
    return xd,yd

save_file="saved/model.ckpt"


X=tf.placeholder("float",[None,128])
Y=tf.placeholder("float",[None,10])

W1=tf.Variable(tf.random_normal([128,5]))
b1=tf.Variable(tf.random_normal([1,5]))

layer1=tf.nn.softmax(tf.matmul(X,W1)+b1)

W2=tf.Variable(tf.random_normal([5,10]))
b2=tf.Variable(tf.random_normal([1,10]))

H=tf.nn.softmax(tf.matmul(layer1,W2)+b2)

r=1e-5
cost=tf.reduce_mean(-tf.reduce_sum(Y*tf.log(H),1))#+r(tf.reduce_sum(tf.square(W1))+tf.reduce_sum(tf.square(W2)))
opt=tf.train.GradientDescentOptimizer(0.1).minimize(cost)

saver=tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #saver.restore(sess,save_file)
    for batch in range(10000):
        x_data,y_data=make_data(2000)
        for step in range(10000):
            ncost,nh,_=sess.run([cost,H,opt],feed_dict={X:x_data,Y:y_data})
            if step%1000==0:
                print("step : ",step,"cost : ",sess.run(tf.reduce_mean(ncost)))
        ac=0
        x_test,y_test=make_data(100)
        for i in range(100):
            ncost,nh,_=sess.run([cost,H,opt],feed_dict={X:x_test[i:i+1],Y:y_test[i:i+1]})
            ans=np.argmax(y_test[i:i+1])
            pred=np.argmax(nh)
            if ans==pred:
                ac+=1
        print("%d/%d : %lf"%(ac,100,ac/100))

    #saver.save(sess,save_file)
