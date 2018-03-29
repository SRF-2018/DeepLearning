import tensorflow as tf
import numpy as np

x_data=np.array([[0,0],[0,1],[1,0],[1,1]],"float")
y_data=np.array([[0],[1],[1],[0]],"float")

X=tf.placeholder("float");
Y=tf.placeholder("float");

W1=tf.Variable(tf.random_normal([2,10]))
b1=tf.Variable(tf.random_normal([10]))

layer1=tf.sigmoid(tf.matmul(X,W1)+b1)

W2=tf.Variable(tf.random_normal([10,1]))
b2=tf.Variable(tf.random_normal([1]))

hypothesis=tf.sigmoid(tf.matmul(layer1,W2)+b2)

cost=-tf.reduce_mean(Y*tf.log(hypothesis)+(1-Y)*tf.log(1-hypothesis))
train=tf.train.GradientDescentOptimizer(0.1).minimize(cost)

w2_hist=tf.summary.histogram("weights2",W2)
cost_summ=tf.summary.scalar("cost",cost)

summary=tf.summary.merge_all()
predicte =tf.cast(hypothesis>0.5,"float")
accuracy =tf.reduce_mean(tf.cast(tf.equal(predicte,Y),"float"))


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    writer=tf.summary.FileWriter('./logs')
    writer.add_graph(sess.graph)
    for step in range(10001):
        s,_=sess.run([summary,train],feed_dict={X:x_data,Y:y_data})
        writer.add_summary(s,global_step=step)
        if step%100==0:
            print(step,sess.run(cost,feed_dict={X:x_data,Y:y_data}))
    h,c,a=sess.run([hypothesis,predicte,accuracy],feed_dict={X:x_data,Y:y_data})
    print("\nHypothesis: ",h,"\nCorrect: ",c,"\nAccuracy: ",a)
