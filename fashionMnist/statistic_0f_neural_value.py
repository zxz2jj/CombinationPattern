import numpy as np
# import pandas as pd


np.set_printoptions(precision=5, suppress=True, threshold=np.nan)


def output_average_std_value(average_list=None, std_list=None, file_path=None):
    if file_path is not None:
        with open(file_path, 'w') as file_to_written:
            if average_list is not None:
                # print("average value:\n", file=file_to_written)
                for average in average_list:
                    print(average, file=file_to_written)

            if std_list is not None:
                print("std value:\n", file=file_to_written)
                for std in std_list:
                    print(std, file=file_to_written)

                print("average + std:\n", file=file_to_written)
                for index in range(len(average_list)):
                    print(average_list[index]+std_list[index], file=file_to_written)

                print("average - std:\n", file=file_to_written)
                for index in range(len(average_list)):
                    print(average_list[index]-std_list[index], file=file_to_written)


for i in range(7):
    with open(r'C:\Users\zz\Myproject\fashionMnist\MNIST_data\training_data\correct_neural_value\fc2\class'+str(i) +
              '_correct_NeuralValue.txt', 'r') as f:
        data = f.read()
        train_correct_neural_value = []
        numlist = data.split()
        for number_str in numlist:
            number_float = float(number_str)
            train_correct_neural_value.append(number_float)
    train_correct_neural_value_transpose = np.transpose(np.array(train_correct_neural_value).reshape([-1, 128]))
    print(train_correct_neural_value_transpose.shape)
    # MaxValue = []
    # VarianceValue = []
    # NumberOfZero = []
    # MinValue = []
    AverageValue = []
    # StdValue = []
    for k in range(128):
        # MaxValue.append(np.max(correctNeuralValue[k]))
        # MinValue.append(np.min(correctNeuralValue[k]))
        AverageValue.append(np.average(train_correct_neural_value_transpose[k]))
        # StdValue.append(np.std(train_correct_neural_value_transpose[k]))
        # VarianceValue.append(np.var(correctNeuralValue[k]))
        # NumberOfZero.append(list(correctNeuralValue[k]).count(0))

    output_path = r'C:\Users\zz\Myproject\fashionMnist\MNIST_data\training_data\correct_neural_value\fc2\class' +\
                  str(i) + '_average.txt'
    output_average_std_value(average_list=AverageValue, file_path=output_path)

    # 阈值为上界：平均值+ rate*方差，下界：平均值-rate*方差  rate [0,1]
    # rate = 1.0
    # UpperThreshold = []
    # LowerThreshold = []
    # for m in range(1024):
    #     UpperThreshold.append(AverageValue[m] + rate * StdValue[m])
    #     LowerThreshold.append(AverageValue[m] - rate * StdValue[m])
    #
    # fw_s = open(r'C:\Users\Myproject\LeNet5\MNIST_data\testing_data\wrong_neural_value\statistic_wrong_prediction_to_['
    #             + str(i) + '].txt', 'w')

    # 统计测试集正确预测的超出阈值百分比
    # with open(r'C:\Users\Myproject\LeNet5\MNIST_data\testing_data\correct_neural_value\class'+str(i) +
    #           '_correct_NeuralValue.txt', 'r') as f:
    #     data = f.read()
    #     test_correctneuralValue = []
    #     numlist = data.split()
    #     for number_str in numlist:
    #         number_float = float(number_str)
    #         test_correctneuralValue.append(number_float)
    # test_correctNeuralValue = np.array(test_correctneuralValue).reshape([-1, 1024])
    #
    # df = pd.DataFrame()
    # df['average'] = AverageValue[:]
    #
    # Sum_reversal = 0.0
    # Sum_num = 0.0
    # Sum_corr = 0.0
    # Sum_distance = 0.0
    #
    # # 记录最大最小神经元异常表达个数
    # Max_num = 0.0
    # Min_num = 100000
    # # 记录最大最小相关系数
    # Max_corr = 0.0
    # Min_corr = 100000
    # # 记录最大最小哈曼顿距离
    # Max_distance = 0.0
    # Min_distance = 100000
    # # 记录最大最小翻转次数
    # Max_reversal = 0.0
    # Min_reversal = 100000

    # 分段记录神经元异常个数
    # number_of_subsection_100 = [0 for i in range(50)]
    # number_of_subsection_50 = [0 for i in range(50)]
    # number_of_subsection_20 = [0 for i in range(50)]
    # number_of_subsection_10 = [0 for i in range(50)]
    # number_of_subsection_5 = [0 for i in range(50)]

    # 分段记录哈曼顿距离个数
    # distance_of_subsection_20 = [0 for i in range(50)]
    # distance_of_subsection_10 = [0 for i in range(50)]
    # number_of_subsection_5 = [0 for i in range(50)]

    # 综合得分记录
    # sum_score = 0.0
    # score = [0 for i in range(70)]
    #
    # for k in range(len(test_correctNeuralValue)):
    #     count = 0
    #     current_distance = 0.0
    #     current_reversal = 0.0
    #     df['example'] = test_correctNeuralValue[k]
    #     for j in range(len(test_correctNeuralValue[0])):
    #         current_distance += abs(test_correctNeuralValue[k][j]-AverageValue[j])
    #         if test_correctNeuralValue[k][j] > UpperThreshold[j] or test_correctNeuralValue[k][j] < LowerThreshold[j]:
    #             count += 1
    #         if j != 0:
    #             if UpperThreshold[j] < UpperThreshold[j-1] and test_correctNeuralValue[k][j] > test_correctNeuralValue[k][j-1]:
    #                 current_reversal += 1
    #             elif UpperThreshold[j] > UpperThreshold[j-1] and test_correctNeuralValue[k][j] < test_correctNeuralValue[k][j-1]:
    #                 current_reversal += 1
    #             else:
    #                 pass
    #
    #     Sum_reversal += current_reversal
    #     if Max_reversal < current_reversal:
    #         Max_reversal = current_reversal
    #     elif Min_reversal > current_reversal:
    #         Min_reversal = current_reversal
    #
    #     number_of_subsection_100[count // 100] += 1
    #     number_of_subsection_50[count // 50] += 1
    #     number_of_subsection_20[count // 20] += 1
    #     number_of_subsection_10[count // 10] += 1
    #     # number_of_subsection_5[count // 5] += 1
    #     Sum_num += count
    #     if Max_num < count:
    #         Max_num = count
    #     elif Min_num > count:
    #         Min_num = count
    #
    #     current_corr = df.corr().get_values()[0][1]
    #     Sum_corr += current_corr
    #     if Max_corr < current_corr:
    #         Max_corr = current_corr
    #     elif Min_corr > current_corr:
    #         Min_corr = current_corr
    #
    #     distance_of_subsection_20[int(current_distance) // 20] += 1
    #     distance_of_subsection_10[int(current_distance) // 10] += 1
    #     Sum_distance += current_distance
    #     if Max_distance < current_distance:
    #         Max_distance = current_distance
    #     elif Min_distance > current_distance:
    #         Min_distance = current_distance
    #
    #     Sc = 0.25 * (count/1024) + 0.25 * (current_distance/1000) + 0.5 * (1 - current_corr)
    #     score[int(Sc/0.01)] += 1
    # print("正确集中神经元超阈值的平均个数:", Sum_num / len(test_correctNeuralValue), "百分比:",
    #       (Sum_num / len(test_correctNeuralValue))/1024, file=fw_s)
    # print("正确集中神经元超阈值的最大值:", Max_num, "最小值：", Min_num,  file=fw_s)
    # print("正确集中与均值的相关系数均值:", Sum_corr/len(test_correctNeuralValue), file=fw_s)
    # print("正确集中与均值的相关系数最大值:", Max_corr, "最小值：", Min_corr,  file=fw_s)
    # print("正确集中哈曼顿距离均值:", Sum_distance/len(test_correctNeuralValue), file=fw_s)
    # print("正确集中哈曼顿距离最大值:", Max_distance, "最小值：", Min_distance,  file=fw_s)
    # print("正确集中翻转均值:", Sum_reversal/len(test_correctNeuralValue), file=fw_s)
    # print("正确集中翻转最大值:", Max_reversal, "最小值：", Min_reversal,  file=fw_s)
    # print("异常神经元个数按区间统计：",  file=fw_s)
    # print("区间数值大小100：\n", number_of_subsection_100,  file=fw_s)
    # print("区间数值大小50：\n", number_of_subsection_50,  file=fw_s)
    # print("区间数值大小20：\n", number_of_subsection_20,  file=fw_s)
    # print("区间数值大小10：\n", number_of_subsection_10, file=fw_s)
    # # print("区间数值大小5：\n", number_of_subsection_5, file=fw_s)
    # print("哈曼顿距离按区间统计：",  file=fw_s)
    # print("区间数值大小20：\n", distance_of_subsection_20,  file=fw_s)
    # print("区间数值大小10：\n", distance_of_subsection_10, file=fw_s)
    # print("综合得分区间为0.01：\n", score, file=fw_s)
    # # 统计测试集错误预测的超出阈值百分比
    # with open(r'C:\Users\Myproject\LeNet5\MNIST_data\testing_data\wrong_neural_value\wrong_prediction_to_[' + str(i) +
    #           ']_NeuralValue.txt', 'r') as f:
    #     data = f.read()
    #     test_wrongneuralValue = []
    #     numlist = data.split()
    #     for number_str in numlist:
    #         number_float = float(number_str)
    #         test_wrongneuralValue.append(number_float)
    # test_wrongNeuralValue = np.array(test_wrongneuralValue).reshape([-1, 1024])
    #
    # for k in range(len(test_wrongNeuralValue)):
    #     df['example'] = test_wrongNeuralValue[k]
    #     print("第", k, "张错误图片：\n", file=fw_s)
    #     Distance = 0.0
    #     count = 0
    #     current_reversal = 0.0
    #     # 记录异常表达的神经元个数
    #     for j in range(1024):
    #         Distance += abs(test_wrongNeuralValue[k][j]-AverageValue[j])
    #         if test_wrongNeuralValue[k][j] > UpperThreshold[j] or test_wrongNeuralValue[k][j] < LowerThreshold[j]:
    #             count += 1
    #             print("神经元位置：", j, end='', file=fw_s)
    #             print("    此位置UpperThreshold：", UpperThreshold[j], "    此位置LowerThreshold：", LowerThreshold[j],
    #                   end='', file=fw_s)
    #             print("    当前值：", test_wrongNeuralValue[k][j], file=fw_s)
    #         if j != 0:
    #             if UpperThreshold[j] < UpperThreshold[j - 1] and test_correctNeuralValue[k][j] > test_correctNeuralValue[k][j - 1]:
    #                 current_reversal += 1
    #             elif UpperThreshold[j] > UpperThreshold[j - 1] and test_correctNeuralValue[k][j] < test_correctNeuralValue[k][j - 1]:
    #                 current_reversal += 1
    #             else:
    #                 pass
    #     print("神经元超阈值个数：", count,  "占总数的百分比：", count/1024, file=fw_s)
    #     print("与均值的相关系数：", df.corr().get_values()[0][1], file=fw_s)
    #     print("哈曼顿距离：", Distance, file=fw_s)
    #     print("翻转次数：", current_reversal, file=fw_s)
    #     print("综合得分：", 0.25 * (count/1024) + 0.25 * (Distance/1000) + 0.5 * (1 - df.corr().get_values()[0][1]), file=fw_s)
    #     print("\n\n", file=fw_s)

