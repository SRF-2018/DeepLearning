import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py
import tensorflow as tf


filename_queue = tf.train.string_input_producer(['input.csv'],shuffle=False,name='filename_queue')

reader = tf.TextLineReader()
key,value = reader.read(filename_queue)

record_defaults = [[0.],[0.],[0.],[0.]]
xy = tf.decode_csv(value,record_defaults=record_defaults)

train_x_data,train_y_data = tf.train.batch([xy[0:-1],xy[-1:]],batch_size=10)


X = tf.placeholder(tf.float32,shape=[None,3])
Y = tf.placeholder(tf.float32,shape=[None,1])

W = tf.Variable(tf.random_normal([3,1]),name='weight')
b = tf.Variable(tf.random_normal([1]),name='bias')

hypothesis = tf.matmul(X,W) + b

cost = tf.reduce_mean(tf.square(hypothesis - Y))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-5)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)
for step in range(2001):
    x_data,y_data = sess.run([train_x_data,train_y_data])
    cost_val,hy_val,_ = sess.run([cost,hypothesis,train],feed_dict={X:x_data,Y:y_data})
    if step % 10 == 0:
        print(step,"Cost: ",cost_val,"\nPrediction:\n",hy_val)


# Ask
print("Your score will be ",sess.run(hypothesis,feed_dict={X:[[100,70,101]]}))

coord.request_stop()
coord.join(threads)