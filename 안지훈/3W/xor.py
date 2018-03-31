import tensorflow as tf
import numpy as np
x_data = np.array([[0,0],[0,1],[1,0],[1,1]],dtype=np.float32)
y_data = np.array([[0]  ,[1]  ,[1]  ,[0]  ],dtype=np.float32)

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
W1 = tf.Variable(tf.random_normal([2,2]),name='weight1')
b1 = tf.Variable(tf.random_normal([2]),name='bias1')
layer1 = tf.sigmoid(tf.matmul(X, W1)+b1)
 
W2 = tf.Variable(tf.random_normal([2,1]),name='weight2')
b2 = tf.Variable(tf.random_normal([1]),name='bias2')
hypothesis = tf.sigmoid(tf.matmul(layer1, W2)+b2)

cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)
predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y),dtype=tf.float32))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(10001):
        _, co = sess.run([train, cost], feed_dict = {X:x_data,Y:y_data})
        if step % 100 == 0:
            print(step, co, sess.run([W1,W2]))
    h, c, a = sess.run([hypothesis, predicted, accuracy],feed_dict = {X:x_data, Y:y_data})
    print("hypothesis :{}, predict :{}, accuracy :{}".format(h,c,a))