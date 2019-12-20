from sklearn.cluster import KMeans
from sklearn import metrics
from sys import maxsize
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pylab as plt
import numpy as np


def read_combination_code(source_path):
    """
    :param source_path: combination code path
    :return:
    """
    numberOfKind = []
    combination_code = []
    sumOfPicture = 0
    y_label = []
    for _kind in range(10):
        file_path = source_path + r'/class' + str(_kind) + '_combination_code.txt'
        with open(file_path, 'r') as file:
            data = file.read()
            data = data.replace('[', ' ').replace(']', ' ').replace(',', ' ')
            numlist = data.split()
            for number_str in numlist:
                number_float = float(number_str)
                combination_code.append(number_float)
            numberOfKind.append(int(len(combination_code) / 256 - sumOfPicture))
            for num in range(numberOfKind[_kind]):
                y_label.append(int(_kind))
            sumOfPicture = len(combination_code) / 256
    combination_code = np.array(combination_code).reshape(-1, 256)

    return sumOfPicture, numberOfKind, combination_code, y_label


def t_sne_visualization(combination_pattern, colour_label, size):
    """
    tsne二维可视化
    :param combination_pattern:
    :param colour_label: 可视化样本点颜色
    :param size: 可视化样本点大小
    :return:
    """
    embedded = TSNE(n_components=2).fit_transform(combination_pattern)
    x_min, x_max = np.min(embedded, 0), np.max(embedded, 0)
    embedded = embedded / (x_max - x_min)
    fig = plt.scatter(embedded[:, 0], embedded[:, 1],
                      c=(np.array(colour_label)/10.0), s=size)
    plt.axis('off')
    plt.show()


def k_means(combination_pattern, clusters, number_of_example, number_by_class, save_path):
    """
    对组合模式求均值并计算各类指标
    :param combination_pattern:
    :param clusters:
    :param number_of_example:
    :param number_by_class:
    :param save_path:
    :return:
    """
    estimator = KMeans(n_clusters=clusters)
    estimator.fit(combination_pattern)
    y_predict = estimator.predict(combination_pattern)
    centers = estimator.cluster_centers_

    real_label = []
    y = [0 for i in range(int(number_of_example))]
    end = 0
    for i in range(10):
        start = end
        end += int(number_by_class[i])
        clusterKind = max(set(y_predict[start:end]), key=list(y_predict[start:end]).count)
        real_label.append(clusterKind)
        for j in range(start, end):
            y[j] = clusterKind
    # scScore = silhouette_score(combinationCode, y_predict)
    homo_Score = metrics.homogeneity_score(y, y_predict)
    comp_Score = metrics.completeness_score(y, y_predict)
    v_measure_ = metrics.v_measure_score(y, y_predict)

    with open(save_path + "/clusterResult_" + ".txt", 'w') as f:
        print(y_predict, '\n', file=f)
        # print("scScore:", scScore, '\n', file=f)
        print("homoScore:", homo_Score, '\n', file=f)
        print("compScore:", comp_Score, '\n', file=f)
        print("v_measure:", v_measure_, '\n', file=f)

    return centers, homo_Score, comp_Score, v_measure_


if __name__ == "__main__":

    np.set_printoptions(suppress=True, threshold=maxsize, precision=2)

    corr_source_path = r"MNIST_data/testing_data/correct_neural_value/fc1/0.05"
    wrong_source_path = r"MNIST_data/testing_data/wrong_neural_value/fc1/0.05"
    wrong_to_source_path = r"MNIST_data/testing_data/wrong_to_neural_value/fc1/0.05"

    corr_sumOfPicture, corr_numberOfKind, corr_combinationCode, corr_y_label = \
        read_combination_code(corr_source_path)

    wrong_sumOfPicture, wrong_numberOfKind, wrong_combinationCode, wrong_y_label = \
        read_combination_code(wrong_source_path)

    wrong_to_sumOfPicture, wrong_to_numberOfKind, wrong_to_combinationCode, wrong_to_y_label = \
        read_combination_code(wrong_to_source_path)

    # print(type(corr_y_label))

    # 将测试集的所有正确样本path进行t-SNE降维并展示（2维）
    print(len(corr_combinationCode))
    t_sne_visualization(combination_pattern=corr_combinationCode, colour_label=corr_y_label, size=1)
    #
    # # 将测试集的所有样本path进行t-SNE降维并展示（2维），相同原始类的样本为同一颜色
    # combinationCode_cw = list(corr_combinationCode) + list(wrong_combinationCode)
    # label_cw = corr_y_label + wrong_y_label
    # s_cw = []
    # for _ in range(int(corr_sumOfPicture)):
    #     s_cw.append(0.1)
    # for _ in range(int(wrong_sumOfPicture)):
    #     s_cw.append(20)
    # t_sne_visualization(combination_pattern=combinationCode_cw, colour_label=label_cw, size=s_cw)
    #
    # # 将测试集的所有样本path进行t-SNE降维并展示（2维），相同原始类的样本为同一颜色
    # combinationCode_cwt = list(corr_combinationCode) + list(wrong_to_combinationCode)
    # label = list(corr_y_label) + list(wrong_to_y_label)
    # s_cwt = []
    # for _ in range(int(corr_sumOfPicture)):
    #     s_cwt.append(0.1)
    # for _ in range(int(wrong_to_sumOfPicture)):
    #     s_cwt.append(20)
    # t_sne_visualization(combination_pattern=combinationCode_cwt, colour_label=label, size=s_cwt)

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

    # 聚类算法
    # for times in range(10):
    cluster_center, homoScore, compScore, v_measure = k_means(corr_combinationCode, 10,
                                                              corr_sumOfPicture, corr_numberOfKind, corr_source_path)
    print(homoScore, compScore, v_measure)


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


