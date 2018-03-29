import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# make training data

num_points=2000
vectors_set=[]

for i in range(num_points):
    if np.random.random()>0.5:
        vectors_set.append([np.random.normal(0.0,0.9),np.random.normal(0.0,0.9),1])
    else:
        vectors_set.append([np.random.normal(3.0,0.5),np.random.normal(1.0,0.5),0])

# df = pd.DataFrame({"x":[v[0]for v in vectors_set],"y":[v[1] for v in vectors_set]})
# sns.lmplot("x","y",data=df,fit_reg=False,size=6)
# plt.show()

xy=np.array(vectors_set)
x_data=xy[:,0:-1]
y_data=xy[:,[-1]]

X=tf.placeholder("float",shape=[None,2])
Y=tf.placeholder("float",shape=[None,1])

W=tf.Variable(tf.random_normal([2,1]),name="weight")
b=tf.Variable(tf.random_normal([1]),name="bias")

hypothesis=tf.sigmoid(tf.matmul(X,W)+b)

cost=-tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))
train=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(2001):
        hyval,coval,_=sess.run([hypothesis,cost,train],feed_dict={X:x_data,Y:y_data})
        vec=[]
        if step%100==0:
            print("cost mean : %lf"%sess.run(tf.reduce_mean(coval)))
            for h, x in zip(hyval,x_data):
                if h>0.5:
                    vec.append([x[0],x[1],1])
                else:
                    vec.append([x[0],x[1],0])
            ndf = pd.DataFrame({"x":[v[0]for v in vec],"y":[v[1] for v in vec],"z":[v[2]for v in vec]})
            sns.lmplot("x","y",hue="z",data=ndf,fit_reg=False,size=6)
            plt.savefig("output/output%02d.png"%(step/100))
            plt.close()
            #plt.show()


