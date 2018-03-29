import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py

import tensorflow as tf
print("LR4")

x_data=[1,2,3]
y_data=[1,2,3]

W = tf.Variable(tf.random_normal([1]),name='weight')
X=tf.placeholder(tf.float32)
Y=tf.placeholder(tf.float32)

#w*x + b
hypothesis = X * W

# cost/Loss function
cost = tf.reduce_mean(tf.square(hypothesis - Y))

# minimize
learning_rate=0.1
gradient=tf.reduce_mean((W*X-Y)*X)
descent=W-learning_rate*gradient
update=W.assign(descent)

# launch session
sess = tf.Session()

#initializes global variables
sess.run(tf.global_variables_initializer())

#fit the line
for step in range(21):
    sess.run(update,feed_dict={X:x_data,Y:y_data})
    print(step,sess.run(cost,feed_dict={X:x_data,Y:y_data}),sess.run(W))

## Testing our model
#print("Test")
#print(sess.run(hypothesis,feed_dict={X:[5]}))
#print(sess.run(hypothesis,feed_dict={X:[2.5]}))
#print(sess.run(hypothesis,feed_dict={X:[1.5,3.5]}))