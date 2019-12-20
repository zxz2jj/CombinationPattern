from cluster import read_combination_code


# 测试正确模式的重叠情况
def reshape_list(combination_pattern, numberOfKind_list):
    """
    将combinationCode按照类别数量进行划分为10个子列表
    :param combination_pattern:
    :param numberOfKind_list:
    :return:
    """
    list_by_kind = []
    end = 0
    for k in range(10):
        start = end
        end += numberOfKind_list[k]
        list_temp = combination_pattern[start:end]
        list_by_kind.append(list_temp)

    return list_by_kind


def set_of_pattern(pattern_list):
    """
    将列表中的模式，将类别中重复模式去除
    :param pattern_list:
    :return: 去重后的模式列表，每一类去重后模式数量
    """
    print("正在进行模式去重...")
    pattern_set = []
    number_of_pattern = []
    c = 0
    for pattern_by_class in pattern_list:
        print("去重：", c)
        c += 1
        temp = []
        for pattern in pattern_by_class:
            pattern = list(pattern)
            if pattern not in temp:
                temp.append(pattern)
        number_of_pattern.append(len(temp))
        pattern_set.append(temp)

    print("模式去重完成！")
    return pattern_set, number_of_pattern


if __name__ == "__main__":

    train_corr_sourcePath = r"MNIST_data/training_data/correct_neural_value/fc1/0.05"
    train_corr_sumOfPicture, train_corr_numberOfKind, train_corr_combinationCode, train_corr_y_label = \
        read_combination_code(train_corr_sourcePath)
    train_corr_pattern_by_kind = reshape_list(train_corr_combinationCode, train_corr_numberOfKind)
    train_corr_pattern_by_kind, new_train_corr_numberOfKind = set_of_pattern(train_corr_pattern_by_kind)
    print(train_corr_numberOfKind)
    print(new_train_corr_numberOfKind)

    print("训练集模式类间查重...")
    savePath1 = train_corr_sourcePath + r'/训练集正确模式重叠情况.txt'
    with open(savePath1, 'w') as f:
        for i in range(10):
            print("查重：", i)
            for j in range(i+1, 10):
                for p in train_corr_pattern_by_kind[i]:
                    if p in train_corr_pattern_by_kind[j]:
                        print("第", i, "类和第", j, "类有重复pattern", '\n', p, file=f)

    test_corr_sourcePath = r"MNIST_data/testing_data/correct_neural_value/fc1/0.05"
    _ts_sum, test_corr_numberOfKind, test_corr_combinationCode, _ts_y = read_combination_code(test_corr_sourcePath)
    test_corr_pattern_by_kind = reshape_list(test_corr_combinationCode, test_corr_numberOfKind)
    test_corr_pattern_by_kind, new_test_corr_numberOfKind = set_of_pattern(test_corr_pattern_by_kind)
    print(test_corr_numberOfKind)
    print(new_test_corr_numberOfKind)
    print("测试集正确模式类间查重...")
    savePath2 = test_corr_sourcePath + r'/测试集正确模式重叠情况.txt'
    with open(savePath2, 'w') as f:
        for i in range(10):
            print("查重：", i)
            for j in range(i+1, 10):
                for p in test_corr_pattern_by_kind[i]:
                    p = list(p)
                    if p in test_corr_pattern_by_kind[j]:
                        print("第", i, "类和第", j, "类有重复pattern", '\n', p, file=f)

    print("测试集正确模式与训练集正确模式匹配...")
    savePath3 = test_corr_sourcePath + r'/测试集正确模式与训练集正确模式匹配情况.txt'
    wrong_match_number = []
    with open(savePath3, 'w') as f:
        for i in range(10):
            print("匹配类：", i)
            count = 0
            for p in test_corr_pattern_by_kind[i]:
                p = list(p)
                if p not in train_corr_pattern_by_kind[i]:
                    count += 1
                    print("第", i, "类存在测试集正确pattern不在训练集正确pattern中：", '\n', p, file=f)
            wrong_match_number.append(count)
            print("第", i, "类共存在测试集正确pattern不在训练集正确pattern中的数量为：", '\n', count, file=f)
    print(wrong_match_number)
    #
    # wrong_sumOfPicture, wrong_numberOfKind, wrong_combinationCode, wrong_y_label = \
    #     read_combination_code(wrong_sourcePath)
    #
    # wrong_to_sumOfPicture, wrong_to_numberOfKind, wrong_to_combinationCode, wrong_to_y_label = \
    #     read_combination_code(wrong_to_sourcePath)





    # wrong_to_combinationCode_by_kind = reshape_list(wrong_to_combinationCode, wrong_numberOfKind)
    #

    #
    # savePath2 = corr_sourcePath + r'/错误模式与正确模式的重叠情况.txt'
    # with open(savePath2, 'w') as f:
    #     for i in range(10):
    #         for j in range(10):
    #             for pattern in wrong_to_combinationCode_by_kind[i]:
    #                 if pattern in corr_combinationCode_by_kind[j]:
    #                     print("第", j, "类错误样本预测模式和第", i, "类正确模式有重复pattern", '\n', pattern, file=f)
