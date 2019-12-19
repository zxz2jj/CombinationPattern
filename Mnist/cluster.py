# from sklearn.cluster import KMeans
# from sklearn import metrics
# from sys import maxsize
# from sklearn.manifold import TSNE
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pylab as plt
import numpy as np


def read_combination_code(sourcePath):
    """
    :param sourcePath: combination code path
    :return:
    """
    numberOfKind = [0]
    combination_code = []
    sumOfPicture = 0
    y_label = []
    for _kind in range(10):
        filepath = sourcePath + r'/class' + str(_kind) + '_combination_code.txt'
        with open(filepath, 'r') as file:
            data = file.read()
            data = data.replace('[', ' ').replace(']', ' ').replace(',', ' ')
            numlist = data.split()
            for number_str in numlist:
                number_float = float(number_str)
                combination_code.append(number_float)
            numberOfKind.append(len(combination_code) / 256 - sumOfPicture)
            for num in range(int(numberOfKind[_kind + 1])):
                y_label.append(float(_kind))
            sumOfPicture = len(combination_code) / 256
    combination_code = np.array(combination_code).reshape(-1, 256)

    return sumOfPicture, numberOfKind, combination_code, y_label


if __name__ == "__main__":
    print("aaaaaaaaaa")
    np.set_printoptions(suppress=True, threshold=maxsize, precision=2)

    corr_sourcePath = r"/Users/zz/PycharmProjects/Mnist/MNIST_data/testing_data/correct_neural_value/fc1/0.4"
    wrong_sourcePath = r"/Users/zz/PycharmProjects/Mnist/MNIST_data/testing_data/wrong_neural_value/fc1/0.4"
    wrong_to_sourcePath = r"/Users/zz/PycharmProjects/Mnist/MNIST_data/testing_data/wrong_to_neural_value/fc1/0.4"

    corr_sumOfPicture, corr_numberOfKind, corr_combinationCode, corr_y_label = \
        read_combination_code(corr_sourcePath)

    wrong_sumOfPicture, wrong_numberOfKind, wrong_combinationCode, wrong_y_label = \
        read_combination_code(wrong_sourcePath)

    wrong_to_sumOfPicture, wrong_to_numberOfKind, wrong_to_combinationCode, wrong_to_y_label = \
        read_combination_code(wrong_to_sourcePath)

    print(len(corr_combinationCode))


# 将测试集数据进行t-SNE降维并展示（3维）
# embedded = TSNE(n_components=3).fit_transform(corr_combinationCode)
# # # 对数据进行归一化操作
# # x_min, x_max = np.min(embedded, 0), np.max(embedded, 0)
# # embedded = embedded / (x_max - x_min)
# # fig = plt.figure()
# # ax = Axes3D(fig)
# # # 将数据对应坐标输入到figure中，不同标签取不同的颜色，共十个类
# # ax.scatter(embedded[:, 0], embedded[:, 1], embedded[:, 2],
# #            c=(np.array(corr_y_label)/10.0))
# # plt.axis('off')
# # plt.show()

# #############################################################################
# # 将测试集的所有正确样本path进行t-SNE降维并展示（2维）
# combinationCode = corr_combinationCode
# label = list(corr_y_label)
# embedded = TSNE(n_components=2).fit_transform(combinationCode)
# x_min, x_max = np.min(embedded, 0), np.max(embedded, 0)
# embedded = embedded / (x_max - x_min)
# fig = plt.scatter(embedded[:, 0], embedded[:, 1],
#                   c=(np.array(label)/10.0), s=1)
# plt.axis('off')
# plt.show()
#
#
# #############################################################################
# # 将测试集的所有样本path进行t-SNE降维并展示（2维），相同原始类的样本为同一颜色
# combinationCode = list(corr_combinationCode) + list(wrong_combinationCode)
# label = list(corr_y_label) + list(wrong_y_label)
# embedded = TSNE(n_components=2).fit_transform(combinationCode)
# x_min, x_max = np.min(embedded, 0), np.max(embedded, 0)
# embedded = embedded / (x_max - x_min)
# s = []
# for _ in range(int(corr_sumOfPicture)):
#     s.append(0.1)
# for _ in range(int(wrong_sumOfPicture)):
#     s.append(20)
# fig = plt.scatter(embedded[:, 0], embedded[:, 1],
#                   c=(np.array(label)/10.0), s=s)
# plt.axis('off')
# plt.show()
#
# #############################################################################
# # 将测试集的所有样本path进行t-SNE降维并展示（2维），相同原始类的样本为同一颜色
# combinationCode = list(corr_combinationCode) + list(wrong_to_combinationCode)
# label = list(corr_y_label) + list(wrong_to_y_label)
# embedded = TSNE(n_components=2).fit_transform(combinationCode)
# x_min, x_max = np.min(embedded, 0), np.max(embedded, 0)
# embedded = embedded / (x_max - x_min)
# s = []
# for _ in range(int(corr_sumOfPicture)):
#     s.append(0.1)
# for _ in range(int(wrong_to_sumOfPicture)):
#     s.append(20)
# fig = plt.scatter(embedded[:, 0], embedded[:, 1],
#                   c=(np.array(label)/10.0), s=s)
# plt.axis('off')
# plt.show()


# ##################################################################################
# 聚类算法
# for times in range(10):
# combinationCode = corr_combinationCode
# estimator = KMeans(n_clusters=10)
# estimator.fit(combinationCode)
# y_predict = estimator.predict(combinationCode)
# centers = estimator.cluster_centers_
# print("聚类中心\n")
# print(centers)

# 根据聚类结果给的分类标签，重新构造对应标签
# real_label = []
# y = [0 for i in range(int(corr_sumOfPicture))]
# start = 0
# end = 0
# for i in range(10):
#     start += int(corr_numberOfKind[i])
#     end += int(corr_numberOfKind[i+1])
#     clusterKind = max(set(y_predict[start:end]), key=list(y_predict[start:end]).count)
#     real_label.append(clusterKind)
#     for j in range(start, end):
#         y[j] = clusterKind
# # scScore = silhouette_score(combinationCode, y_predict)
# homoScore = metrics.homogeneity_score(y, y_predict)
# compScore = metrics.completeness_score(y, y_predict)
# v_measure = metrics.v_measure_score(y, y_predict)
# print(homoScore, compScore, v_measure)
# print(real_label)
#
# with open(corr_sourcePath + "/clusterResult_" + ".txt", 'w') as f:
#     print(y_predict, '\n', file=f)
#     # print("scScore:", scScore, '\n', file=f)
#     print("homoScore:", homoScore, '\n', file=f)
#     print("compScore:", compScore, '\n', file=f)
#     print("v_measure:", v_measure, '\n', file=f)

###########################################################################
# 计算距离
# real_center = []
# for kind in range(10):
#     real_center.append(list(centers[real_label[kind]]))
# print("调整后中心\n")
# print(np.array(real_center))
#
# corr_max_list = []
# corr_min_list = []
# corr_avg_list = []
# for kind in range(1, 11):
#     corr_max_distance = 0.0
#     corr_min_distance = 99999.0
#     corr_sum_distance = 0.0
#     corr_avg_distance = 0.0
#     index = int(sum(corr_numberOfKind[0:kind]))
#     # print(index)
#     for i in range(int(corr_numberOfKind[kind])):
#         distance = 0.0
#         for k in range(256):
#             distance += abs(corr_combinationCode[index + i][k] - real_center[kind-1][k])
#         corr_sum_distance += distance
#         if distance > corr_max_distance:
#             corr_max_distance = distance
#         if distance < corr_min_distance:
#             corr_min_distance = distance
#     corr_max_list.append(corr_max_distance)
#     corr_min_list.append(corr_min_distance)
#     corr_avg_list.append(corr_sum_distance / corr_numberOfKind[kind])
# print(corr_max_list)
# print(corr_min_list)
# print(corr_avg_list)
#
# # 错误样本在原始类上的组合模式与原始类别的类中心距离
# wrong_max_list = []
# wrong_min_list = []
# wrong_avg_list = []
# for kind in range(1, 11):
#     wrong_max_distance = 0.0
#     wrong_min_distance = 99999.0
#     wrong_sum_distance = 0.0
#     wrong_avg_distance = 0.0
#     index = int(sum(wrong_numberOfKind[0:kind]))
#     # print(index)
#     for i in range(int(wrong_numberOfKind[kind])):
#         distance = 0.0
#         for k in range(256):
#             distance += abs(wrong_combinationCode[index + i][k] - real_center[kind-1][k])
#         wrong_sum_distance += distance
#         if distance > wrong_max_distance:
#             wrong_max_distance = distance
#         if distance < wrong_min_distance:
#             wrong_min_distance = distance
#     wrong_max_list.append(wrong_max_distance)
#     wrong_min_list.append(wrong_min_distance)
#     wrong_avg_list.append(wrong_sum_distance / wrong_numberOfKind[kind])
# print(wrong_max_list)
# print(wrong_min_list)
# print(wrong_avg_list)
#
#
# # 错误预测样本在预测类别上的组合编码与预测类中心的距离
# wrong_to_max_list = []
# wrong_to_min_list = []
# wrong_to_avg_list = []
# for kind in range(1, 11):
#     wrong_to_max_distance = 0.0
#     wrong_to_min_distance = 99999.0
#     wrong_to_sum_distance = 0.0
#     wrong_to_avg_distance = 0.0
#     index = int(sum(wrong_to_numberOfKind[0:kind]))
#     # print(index)
#     for i in range(int(wrong_to_numberOfKind[kind])):
#         distance = 0.0
#         for k in range(256):
#             distance += abs(wrong_to_combinationCode[index + i][k] - real_center[kind-1][k])
#         wrong_to_sum_distance += distance
#         if distance > wrong_to_max_distance:
#             wrong_to_max_distance = distance
#         if distance < wrong_to_min_distance:
#             wrong_to_min_distance = distance
#     wrong_to_max_list.append(wrong_to_max_distance)
#     wrong_to_min_list.append(wrong_to_min_distance)
#     wrong_to_avg_list.append(wrong_to_sum_distance / wrong_to_numberOfKind[kind])
# print(wrong_to_max_list)
# print(wrong_to_min_list)
# print(wrong_to_avg_list)


