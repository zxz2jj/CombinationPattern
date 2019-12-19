from sklearn.cluster import KMeans
from sklearn import metrics
from sys import maxsize
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pylab as plt
import numpy as np

np.set_printoptions(suppress=True, threshold=maxsize, precision=2)

corr_sourcePath = r"/Users/zz/PycharmProjects/fashionMnist/MNIST_data/testing_data/correct_neural_value/fc1/0.4"
wrong_sourcePath = r"/Users/zz/PycharmProjects/fashionMnist/MNIST_data/testing_data/wrong_neural_value/fc1/0.4"
wrong_to_sourcePath = r"/Users/zz/PycharmProjects/fashionMnist/MNIST_data/testing_data/wrong_to_neural_value/fc1/0.4"


corr_numberOfKind = [0]
corr_combinationCode = []
corr_sumOfPicture = 0
corr_y_label = []
for kind in range(10):
    filepath = corr_sourcePath + r'/class' + str(kind) + '_combination_code.txt'
    with open(filepath, 'r') as f:
        data = f.read()
        data = data.replace('[', ' ').replace(']', ' ').replace(',', ' ')
        numlist = data.split()
        for number_str in numlist:
            number_float = float(number_str)
            corr_combinationCode.append(number_float)
        corr_numberOfKind.append(len(corr_combinationCode)/256 - corr_sumOfPicture)
        for num in range(int(corr_numberOfKind[kind+1])):
            corr_y_label.append(float(kind))
        corr_sumOfPicture = len(corr_combinationCode)/256
corr_combinationCode = np.array(corr_combinationCode).reshape(-1, 256)
print(len(corr_combinationCode))
print(corr_sumOfPicture)
print(corr_numberOfKind)
print(len(corr_y_label))

wrong_numberOfKind = [0]
wrong_combinationCode = []
wrong_sumOfPicture = 0
wrong_y_label = []
for kind in range(10):
    filepath = wrong_sourcePath + r'/class' + str(kind) + '_combination_code.txt'
    with open(filepath, 'r') as f:
        data = f.read()
        data = data.replace('[', ' ').replace(']', ' ').replace(',', ' ')
        numlist = data.split()
        for number_str in numlist:
            number_float = float(number_str)
            wrong_combinationCode.append(number_float)
        wrong_numberOfKind.append(len(wrong_combinationCode)/256 - wrong_sumOfPicture)
        for num in range(int(wrong_numberOfKind[kind+1])):
            wrong_y_label.append(kind)
        wrong_sumOfPicture = len(wrong_combinationCode)/256
wrong_combinationCode = np.array(wrong_combinationCode).reshape(-1, 256)
print(len(wrong_combinationCode))
print(wrong_sumOfPicture)
print(wrong_numberOfKind)
print(len(wrong_y_label))

wrong_to_numberOfKind = [0]
wrong_to_combinationCode = []
wrong_to_sumOfPicture = 0
wrong_to_y_label = []
for kind in range(10):
    filepath = wrong_to_sourcePath + r'/class' + str(kind) + '_combination_code.txt'
    with open(filepath, 'r') as f:
        data = f.read()
        data = data.replace('[', ' ').replace(']', ' ').replace(',', ' ')
        numlist = data.split()
        for number_str in numlist:
            number_float = float(number_str)
            wrong_to_combinationCode.append(number_float)
        wrong_to_numberOfKind.append(len(wrong_to_combinationCode)/256 - wrong_to_sumOfPicture)
        for num in range(int(wrong_to_numberOfKind[kind+1])):
            wrong_to_y_label.append(kind)
        wrong_to_sumOfPicture = len(wrong_to_combinationCode)/256
wrong_to_combinationCode = np.array(wrong_to_combinationCode).reshape(-1, 256)
print(len(wrong_to_combinationCode))
print(wrong_to_sumOfPicture)
print(wrong_to_numberOfKind)
print(len(wrong_to_y_label))

# # 将测试集数据进行t-SNE降维并展示（3维）
# embedded = TSNE(n_components=3).fit_transform(combinationCode)
# # 对数据进行归一化操作
# x_min, x_max = np.min(embedded, 0), np.max(embedded, 0)
# embedded = embedded / (x_max - x_min)
# # 创建显示的figure
# fig = plt.figure()
# ax = Axes3D(fig)
# # 将数据对应坐标输入到figure中，不同标签取不同的颜色，共十个类
# ax.scatter(embedded[:, 0], embedded[:, 1], embedded[:, 2],
#            c=(np.array(label)/10.0))
# plt.axis('off')
# plt.show()


# 将测试集数据进行t-SNE降维并展示（2维）
# combinationCode = corr_combinationCode
# print(len(combinationCode))
# # label = list(corr_y_label) + list(wrong_to_y_label)
# # print(len(label))
# estimator = KMeans(n_clusters=10)
# estimator.fit(combinationCode)
# y_predict = estimator.predict(combinationCode)

# embedded = TSNE(n_components=2).fit_transform(combinationCode)
# x_min, x_max = np.min(embedded, 0), np.max(embedded, 0)
# embedded = embedded / (x_max - x_min)
# s = []
# for _ in range(int(corr_sumOfPicture)):
#     s.append(0.1)
# for _ in range(int(wrong_to_sumOfPicture)):
#     s.append(20)
# # # 创建显示的figure
# fig = plt.scatter(embedded[:, 0], embedded[:, 1],
#                   c=(np.array(label)/10.0 + 0.3), s=s)
#
# plt.axis('off')
# plt.show()


# for times in range(10):
# 聚类算法
combinationCode = corr_combinationCode
estimator = KMeans(n_clusters=10)
estimator.fit(combinationCode)
y_predict = estimator.predict(combinationCode)
centers = estimator.cluster_centers_
# print("聚类中心\n")
# print(centers)

# 调整类中心顺序
real_label = []
for kind in range(1, 11):
    offset = int(sum(corr_numberOfKind[0:kind]))
    # print(index)
    _distance = []
    for c in range(10):
        distance = 0.0
        for i in range(int(corr_numberOfKind[kind])):
            for k in range(256):
                distance += abs(corr_combinationCode[offset + i][k] - centers[c][k])
        _distance.append(distance)
        # print(_distance)
    real_label.append(_distance.index(min(_distance)))
# print(real_label)

real_center = []
for kind in range(10):
    real_center.append(list(centers[real_label[kind]]))
# print("调整后中心\n")
# print(np.array(real_center))

corr_max_list = []
corr_min_list = []
corr_avg_list = []
for kind in range(1, 11):
    corr_max_distance = 0.0
    corr_min_distance = 99999.0
    corr_sum_distance = 0.0
    corr_avg_distance = 0.0
    index = int(sum(corr_numberOfKind[0:kind]))
    # print(index)
    for i in range(int(corr_numberOfKind[kind])):
        distance = 0.0
        for k in range(256):
            distance += abs(corr_combinationCode[index + i][k] - real_center[kind-1][k])
        corr_sum_distance += distance
        if distance > corr_max_distance:
            corr_max_distance = distance
        if distance < corr_min_distance:
            corr_min_distance = distance
    corr_max_list.append(corr_max_distance)
    corr_min_list.append(corr_min_distance)
    corr_avg_list.append(corr_sum_distance / corr_numberOfKind[kind])
print(corr_max_list)
print(corr_min_list)
print(corr_avg_list)


wrong_max_list = []
wrong_min_list = []
wrong_avg_list = []
for kind in range(1, 11):
    wrong_max_distance = 0.0
    wrong_min_distance = 99999.0
    wrong_sum_distance = 0.0
    wrong_avg_distance = 0.0
    index = int(sum(wrong_numberOfKind[0:kind]))
    # print(index)
    for i in range(int(wrong_numberOfKind[kind])):
        distance = 0.0
        for k in range(256):
            distance += abs(wrong_combinationCode[index + i][k] - real_center[kind-1][k])
        wrong_sum_distance += distance
        if distance > wrong_max_distance:
            wrong_max_distance = distance
        if distance < wrong_min_distance:
            wrong_min_distance = distance
    wrong_max_list.append(wrong_max_distance)
    wrong_min_list.append(wrong_min_distance)
    wrong_avg_list.append(wrong_sum_distance / wrong_numberOfKind[kind])
print(wrong_max_list)
print(wrong_min_list)
print(wrong_avg_list)


wrong_to_max_list = []
wrong_to_min_list = []
wrong_to_avg_list = []
for kind in range(1, 11):
    wrong_to_max_distance = 0.0
    wrong_to_min_distance = 99999.0
    wrong_to_sum_distance = 0.0
    wrong_to_avg_distance = 0.0
    index = int(sum(wrong_to_numberOfKind[0:kind]))
    # print(index)
    for i in range(int(wrong_to_numberOfKind[kind])):
        distance = 0.0
        for k in range(256):
            distance += abs(wrong_to_combinationCode[index + i][k] - real_center[kind-1][k])
        wrong_to_sum_distance += distance
        if distance > wrong_to_max_distance:
            wrong_to_max_distance = distance
        if distance < wrong_to_min_distance:
            wrong_to_min_distance = distance
    wrong_to_max_list.append(wrong_to_max_distance)
    wrong_to_min_list.append(wrong_to_min_distance)
    wrong_to_avg_list.append(wrong_to_sum_distance / wrong_to_numberOfKind[kind])
print(wrong_to_max_list)
print(wrong_to_min_list)
print(wrong_to_avg_list)

# 根据聚类结果给的分类标签，构造正确标签
y = [0 for i in range(int(corr_sumOfPicture))]
start = 0
end = 0
for i in range(10):
    start += int(corr_numberOfKind[i])
    end += int(corr_numberOfKind[i+1])
    clusterKind = max(set(y_predict[start:end]), key=list(y_predict[start:end]).count)
    for j in range(start, end):
        y[j] = clusterKind
# scScore = silhouette_score(combinationCode, y_predict)
homoScore = metrics.homogeneity_score(y, y_predict)
compScore = metrics.completeness_score(y, y_predict)
v_measure = metrics.v_measure_score(y, y_predict)
print(homoScore, compScore, v_measure)

# with open(corr_sourcePath + "/clusterResult_" + ".txt", 'w') as f:
#     print(y_predict, '\n', file=f)
#     # print(y, '\n', file=f)
#     # print("scScore:", scScore, '\n', file=f)
#     print("homoScore:", homoScore, '\n', file=f)
#     print("compScore:", compScore, '\n', file=f)
#     print("v_measure:", v_measure, '\n', file=f)
