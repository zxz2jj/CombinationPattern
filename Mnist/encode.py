import numpy as np
import math


def load_neural_value(fpath, number_of_neuron):
    """
    根据所给文件把神经元输出值从文件中读取
    :param fpath:
    :param number_of_neuron:
    :return:
    """
    with open(fpath, 'r') as f:
        data = f.read()
        numlist = data.split()
        neural_value_ = []
        for number_str in numlist:
            number_float = float(number_str)
            neural_value_.append(number_float)
    neural_value_ = np.array(neural_value_).reshape([-1, number_of_neuron])

    return neural_value_


def get_neural_value_and_average(source_path, number_of_neuron, number_of_classes):
    """
    读取所有样本的神经元输出值，计算某神经元在非第i类上输出均值
    non_class_l_average[l][k]表示第k个神经元在非第l类上的输出均值
    :param
    :return:  correct_neural_value[]  non_class_l_average[] class_l_average[]
    """
    non_class_l_average = [[] for i in range(number_of_classes)]
    class_l_average = [[] for i in range(number_of_classes)]
    class_l_neuron_sum = [[] for i in range(number_of_classes)]
    number_of_examples = []
    all_class_average = []

    for i in range(number_of_classes):
        filepath = source_path + r'/class' + str(i) + '_correct_NeuralValue.txt'
        correct_neural_value = load_neural_value(filepath, number_of_neuron)
        number_of_examples.append(len(correct_neural_value))
        train_correct_neural_value_transpose = np.transpose(correct_neural_value)
        # print(number_of_examples)
        for k in range(number_of_neuron):
            class_l_neuron_sum[i].append(np.sum(train_correct_neural_value_transpose[k]))
            class_l_average[i].append(class_l_neuron_sum[i][k] / number_of_examples[i])

    for k in range(number_of_neuron):    # 第k个神经元
        output_sum = 0.0
        for l in range(number_of_classes):    # 第l类
            output_sum += class_l_neuron_sum[l][k]
        all_class_average.append(output_sum / np.sum(number_of_examples))
        for c in range(number_of_classes):    # 对每一类的第k个神经元
            non_class_l_average[c].append((output_sum - class_l_neuron_sum[c][k]) /
                                          (np.sum(number_of_examples) - number_of_examples[c]))
    # print(all_class_average)
    # print(non_class_l_average[0])
    # print(class_l_average[0])

    return non_class_l_average, all_class_average, class_l_average


def encode(image_neural_value,
           label,
           encode_rate,
           number_of_neuron,
           non_class_l_average,
           all_class_average=None):
    """
    根据某selective值对某一图片的神经元进行编码
    :param image_neural_value: 图片的神经元输出
    :param label: 图片的预测标签
    :param number_of_neuron: 神经元个数
    :param encode_rate: 编码率
    :param non_class_l_average: 训练集非l类的图片的神经元输出均值
    :param all_class_average: 训练集所有类的图片的神经元输出均值
    :return: combination_code
    """
    selective = [0.0 for i in range(number_of_neuron)]
    if all_class_average is not None:
        for i in range(number_of_neuron):
            if all_class_average[i] == 0:
                selective[i] = 0
            else:
                selective[i] = (image_neural_value[i] - non_class_l_average[label][i]) / \
                               all_class_average[i]
    else:
        for i in range(number_of_neuron):
            selective[i] = image_neural_value[i] - non_class_l_average[label][i]

    dict_sel = {}
    for index in range(len(selective)):
        dict_sel[index] = selective[index]
    sort_by_sel = sorted(dict_sel.items(), key=lambda x: x[1])
    # print(sort_by_sel)
    combination_code = [0 for i in range(len(selective))]
    for k in range(0, math.ceil(number_of_neuron * encode_rate / 2), 1):
        combination_code[sort_by_sel[k][0]] = -1
    for k in range(0, -math.ceil(number_of_neuron * encode_rate / 2), -1):
        combination_code[sort_by_sel[k][0]] = 1

    # print(list(image_neural_value))
    # print(combination_code)
    return combination_code


if __name__ == "__main__":
    correct_south_path = "MNIST_data/training_data/correct_neural_value/fc1"
    # coding_examples_path = "MNIST_data/training_data/correct_neural_value/fc1"
    # coding_examples_path = "MNIST_data/testing_data/correct_neural_value/fc1"
    coding_examples_path = "MNIST_data/testing_data/wrong_neural_value/fc1"
    # coding_examples_path = "MNIST_data/testing_data/wrong_to_neural_value/fc1"
    non_class_l_avg, all_class_avg, class_l_avg = get_neural_value_and_average(correct_south_path, 256, 10)
    # class_code = []
    for kind in range(10):
        # input_path = coding_examples_path + r'/class' + str(kind) + '_correct_NeuralValue.txt'
        # input_path = coding_examples_path + r'/class' + str(kind) + '_correct_NeuralValue.txt'
        input_path = coding_examples_path + r'/class' + str(kind) + '_wrong_NeuralValue.txt'
        # input_path = coding_examples_path + r'/wrong_to_' + str(kind) + 'NeuralValue.txt'
        output_path = open(coding_examples_path + r'/class' + str(kind) + '_combination_code.txt', 'w')
        neural_value = load_neural_value(input_path, 256)
        for each_image in neural_value:
            pattern = encode(each_image, kind, 0.2, 256, non_class_l_avg)
            print(pattern, file=output_path)
        # class_code.append(encode(class_l_avg[kind], kind, 0.4, 256, non_class_l_avg))
    # print(class_code)
