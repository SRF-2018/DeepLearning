import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

img = mnist.train.images[0].reshape(28, 28)
plt.imshow(img, cmap='gray')
plt.show()

sess = tf.InteractiveSession()
#n개(-1)의 28X26X1 이미지
img = img.reshape(-1, 28, 28, 1)
#3X3X1 크기의 필터가 5개, 표준편차 0.01, 평균은 없으면 0.0 초기값
W1 = tf.Variable(tf.random_normal([3, 3, 1, 5], stddev=0.01))

conv2d = tf.nn.conv2d(img, W1, strides=[1, 2, 2, 1], padding='SAME')
print(conv2d)
sess.run(tf.global_variables_initializer())
conv2d_img = conv2d.eval()
conv2d_img = np.swapaxes(conv2d_img, 0, 3)
for i, one_img in enumerate(conv2d_img):
    plt.subplot(1, 5, i+1), plt.imshow(one_img.reshape(14,14), cmap='gray')
plt.show()