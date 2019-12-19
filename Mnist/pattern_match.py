from cluster import read_combination_code
# from cluster import corr_sourcePath, wrong_sourcePath, wrong_to_sourcePath

train_corr_sourcePath = r"MNIST_data/training_data/correct_neural_value/fc1/0.4"

train_corr_sumOfPicture, train_corr_numberOfKind, train_corr_combinationCode, train_corr_y_label = \
    read_combination_code(train_corr_sourcePath)

# corr_sumOfPicture, corr_numberOfKind, corr_combinationCode, corr_y_label = \
#     read_combination_code(corr_sourcePath)
#
# wrong_sumOfPicture, wrong_numberOfKind, wrong_combinationCode, wrong_y_label = \
#     read_combination_code(wrong_sourcePath)
#
# wrong_to_sumOfPicture, wrong_to_numberOfKind, wrong_to_combinationCode, wrong_to_y_label = \
#     read_combination_code(wrong_to_sourcePath)

print(len(train_corr_combinationCode))
print(train_corr_numberOfKind)


#########################################################################
# 测试正确模式的重叠情况
def reshape_list(combinationCode, numberOfKind_list):
    """
    将combinationCode按照类别数量进行划分为10个子列表
    :param combinationCode:
    :param numberOfKind_list:
    :return:
    """
    list_by_kind = []
    end = 0
    for i in range(1, 11):
        start = end
        end += numberOfKind_list[i]
        list_temp = combinationCode[int(start):int(end)]
        list_by_kind.append(list_temp)

    return list_by_kind


def set_list(list_t):
    """
    将列表list_t中的重复子列表去除
    :param list_t:
    :return:
    """
    temp = []
    for _ in list_t:
        if _ not in temp:
            temp.append(_)

    return temp


corr_combinationCode_by_kind = reshape_list(train_corr_combinationCode, train_corr_numberOfKind)
# wrong_to_combinationCode_by_kind = reshape_list(wrong_to_combinationCode, wrong_numberOfKind)
#
# savePath1 = corr_sourcePath + r'/正确模式重叠情况.txt'
# with open(savePath1, 'w') as f:
#     for i in range(10):
#         for j in range(i+1, 10):
#             for pattern in corr_combinationCode_by_kind[i]:
#                 if pattern in corr_combinationCode_by_kind[j]:
#                     print("第", i, "类和第", j, "类有重复pattern", '\n', pattern, file=f)
#
# savePath2 = corr_sourcePath + r'/错误模式与正确模式的重叠情况.txt'
# with open(savePath2, 'w') as f:
#     for i in range(10):
#         for j in range(10):
#             for pattern in wrong_to_combinationCode_by_kind[i]:
#                 if pattern in corr_combinationCode_by_kind[j]:
#                     print("第", j, "类错误样本预测模式和第", i, "类正确模式有重复pattern", '\n', pattern, file=f)
