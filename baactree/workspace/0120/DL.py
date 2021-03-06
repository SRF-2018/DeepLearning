import tensorflow as tf
import numpy as np
import random
import main

def make_data(num):
    xd=[]
    yd=[]
    for size in range(num):
        main.init()
        number=random.randrange(0,10000)
        main.run(number)
        xd.append(main.photo)
        yd.append(number)
    xd=np.array(xd)
    yd=np.array(yd)
    xd=np.reshape(xd,[-1,16,64,1])
    yd=np.reshape(yd,[-1,1])
    return xd,yd

save_file="saved/model.ckpt"

####
learning_rate=0.001
print("learning_rate : ",learning_rate)
####


X=tf.placeholder(tf.float32,[None,16,64,1])
Y=tf.placeholder(tf.int32,[None,1])
_Y=tf.reshape(tf.one_hot(Y,10000),[-1,10000])
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
flat=tf.reshape(dropout3,[-1,2*8*128])
dense4=tf.layers.dense(inputs=flat,units=625,activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer())
dropout4=tf.layers.dropout(inputs=dense4,rate=0.5,training=training)

# logit 
logits=tf.layers.dense(inputs=dropout4,units=10000,kernel_initializer=tf.contrib.layers.xavier_initializer())

cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=_Y))
optimizer=tf.train.AdamOptimizer(learning_rate).minimize(cost)

is_correct=tf.equal(tf.argmax(logits,1),tf.argmax(_Y,1))
accuracy=tf.reduce_mean(tf.cast(is_correct,"float"))

saver=tf.train.Saver()

##
training_epochs=5

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #saver.restore(sess,save_file)
    for epoch in range(training_epochs):
        avg_cost=0
        total_batch=1000
        for batch in range(total_batch):
            batch_x,batch_y=make_data(500)
            batch_cost,_=sess.run([cost,optimizer],feed_dict={X:batch_x,Y:batch_y,training:True})
            avg_cost+=batch_cost/total_batch
        print("Epoch :","%04d"%(epoch+1),"cost =","{:.9f}".format(avg_cost))
    x_test,y_test=make_data(10000)
    print("Final Accuracy: ",sess.run(accuracy,feed_dict={X:x_test,Y:y_test,training:False}))
    #saver.save(sess,save_file)
