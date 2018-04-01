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

learning_rate=0.001
print(learning_rate)
X=tf.placeholder("float",[None,128])
Y=tf.placeholder("float",[None,10])
keep_prob=tf.placeholder("float")

W1=tf.get_variable("W1",shape=[128,256],initializer=tf.contrib.layers.xavier_initializer())
b1=tf.Variable(tf.random_normal([256]))
L1=tf.nn.relu(tf.matmul(X,W1)+b1)
L1=tf.nn.dropout(L1,keep_prob=keep_prob)

#W2=tf.Variable(tf.random_normal([256,256]))
W2=tf.get_variable("W2",shape=[256,256],initializer=tf.contrib.layers.xavier_initializer())
b2=tf.Variable(tf.random_normal([256]))
L2=tf.nn.relu(tf.matmul(L1,W2)+b2)
L2=tf.nn.dropout(L2,keep_prob=keep_prob)

#W3=tf.Variable(tf.random_normal([256,10]))
W3=tf.get_variable("W3",shape=[256,10],initializer=tf.contrib.layers.xavier_initializer())
b3=tf.Variable(tf.random_normal([10]))
H=tf.matmul(L2,W3)+b3;

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=H,labels=Y))
opt=tf.train.AdamOptimizer(learning_rate).minimize(cost)

saver=tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #saver.restore(sess,save_file)
    for batch in range(10000):
        x_data,y_data=make_data(2000)
        for step in range(10000):
            ncost,nh,_=sess.run([cost,H,opt],feed_dict={X:x_data,Y:y_data,keep_prob:0.7})
            if step%1000==0:
                print("step : ",step,"cost : ",sess.run(tf.reduce_mean(ncost)))
        ac=0
        x_test,y_test=make_data(100)
        for i in range(100):
            ncost,nh,_=sess.run([cost,H,opt],feed_dict={X:x_test[i:i+1],Y:y_test[i:i+1],keep_prob:1})
            ans=np.argmax(y_test[i:i+1])
            pred=np.argmax(nh)
            if ans==pred:
                ac+=1
        print("%d/%d : %lf"%(ac,100,ac/100))

    #saver.save(sess,save_file)
