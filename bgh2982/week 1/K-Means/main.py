# 군집화
# k개 중심의 초기 집합 결정
# 각 데이터를 가장 가까운 군집에 할당
# 각 그룹에 대해 새로운 중심을 계산
import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

num_points = 2000
vectors_set = []

for i in range(num_points):
    if np.random.random() > 0.5:
        vectors_set.append([np.random.normal(0.0, 0.9), np.random.normal(0.0, 0.9)])
    else:
        vectors_set.append([np.random.normal(3.0, 0.5), np.random.normal(1.0, 0.5)])

df = pd.DataFrame({"x":[v[0] for v in vectors_set], "y":[v[1] for v in vectors_set]})

sns.lmplot("x", "y", data=df, fit_reg=False, size=6)
plt.show()

vectors = tf.constant(vectors_set)
k =  4
#k개의 중심 선택
# 2D 텐서로 저장
centroides = tf.Variable(tf.slice(tf.random_shuffle(vectors), [0, 0], [k, -1]))

#print(vectors.get_shape())
#print(centroides.get_shape())

#2000X2 과 4X2를 빼기 위해 3차원으로 늘린다.
#vector(2000X2) -> expanded_vectors(1X2000X2)
#centroides(4X2) -> expanded_centroides(4X1X2)
#tensorflow의 broadcasting에 의해 크기가 1인 차원은 다른 텐서의 해당 차원 크기에 맞게 계산을 반복한다.
expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroides = tf.expand_dims(centroides, 1)

#유클리드 제곱거리를 계산해 가장 작은 중심 계산
#뺄셈 (4X2000X2)
#diff = tf.subtract(expanded_vectors, expanded_centroides)
#제곱(4X2000X2)
#  sqr = tf.square(diff)
#지정 차원의 원소 값들을 더함 (4X2000)
#distances = tf.reduce_sum(sqr, 2)
#지정 차원에서 가장 작은 값의 인덱스 리턴 (2000)
#assignments = tf.argmin(distances, 0)

assignments = tf.argmin(tf.reduce_sum(tf.square(tf.subtract(expanded_vectors, expanded_centroides)), 2), 0)


#한 군집(c)과 매칭되는 assignments 텐서의 각 원소 위치를 true로 표시하는 원소만 true로 표시하는 텐서 생성(2000)
#boolean = tf.equal(assignments, c)
#매개변수로 받은 불리언 텐서에서  true로 표시된 위치의 값으로 가지는 텐서 생성 (2000X1)
#location = tf.where(boolean)
#군집에 속한 vectors 텐서의 포인터들의 인덱스로 구성된 텐서 생성 (1X2000)
#points_index = tf.reshape(location, [1, -1])
#군집을 이루는 점들의 좌표를 모은 텐서 생성(1X2000x2)
#points_location = tf.gather(vectors, points_index)
#군집에 속한 모든 점의 평균 값을 가진 텐서를 생성(1X2)
#tf.reduce_mean(points_location, reduction_indices=[1])
means = tf.concat([tf.reduce_mean(tf.gather(vectors, tf.reshape(tf.where(tf.equal(assignments, c)), [1, -1])), reduction_indices=[1]) for c in range(k)], 0)

update_centroides = tf.assign(centroides, means)

init_op = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init_op)

for step in range(11):
    _, centroid_values, assignment_values = sess.run([update_centroides, centroides, assignments])
    if step%1==0:
        data = {"x": [], "y": [], "cluster": []}

        for i in range(len(assignment_values)):
            data["x"].append(vectors_set[i][0])
            data["y"].append(vectors_set[i][1])
            data["cluster"].append(assignment_values[i])

        df = pd.DataFrame(data)

        sns.lmplot("x", "y", data=df, fit_reg=False, size=6, hue="cluster", legend=False)

        fig = plt.gcf()  # 변경한 곳
        plt.show()
        fig.savefig('k-menas'+str(step+1)+'.png') #변경한 곳
