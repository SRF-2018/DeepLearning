import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
num_points = 2000
vectors_set = []

for i in range(num_points):
    if np.random.random() > 0.5:
        vectors_set.append([np.random.normal(0.0, 0.9), np.random.normal(0.0, 0.9)])
    else:
        vectors_set.append([np.random.normal(3.0, 0.5), np.random.normal(1.0, 0.5)])

vectors=np.array(vectors_set)

k=4

evec=[[v for i in range(k)] for v in vectors]
evec=np.array(evec)

centroides = tf.Variable(tf.slice(tf.random_shuffle(vectors),[0,0],[k,-1]))

assignments=tf.argmin(tf.reduce_sum(tf.square(tf.subtract(evec,centroides)),axis=2),axis=1)

means=tf.concat([tf.reduce_mean(tf.gather(vectors,tf.reshape(tf.where(tf.equal(assignments,i)),[1,-1])),reduction_indices=[1]) for i in range(k)],axis=0)

update_centroids=tf.assign(centroides,means)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(20):
        _,nvec=sess.run([update_centroids,assignments])
        data = {"x":[],"y":[],"cluster":[]}
        for xy, c in zip(vectors,nvec):
            data["x"].append(xy[0])
            data["y"].append(xy[1])
            data["cluster"].append(c)
        df=pd.DataFrame(data)
        sns.lmplot("x", "y", data=df, fit_reg=False, size=6, hue="cluster", legend=False)
        plt.savefig("output/output%02d.png"%step)
        plt.close()
    