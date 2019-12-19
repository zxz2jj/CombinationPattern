import numpy as np
import tensorflow as tf
import sys

np.set_printoptions(precision=8, suppress=True, threshold=sys.maxsize)


def get_neural_value(source_path, result_path, tensor_name, number_of_value):
    """
    给出图片数据集，输出每张图片经过网络后，每个位置上的神经元输出值
    """
    with open(source_path, 'r') as source_file:
        data = source_file.read()
        image = []
        numlist = data.split()
        for number_str in numlist:
            number_float = float(number_str)
            image.append(number_float)
    images = np.array(image).reshape([-1, 784])
    label = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    sess = tf.Session()
    saver = tf.train.import_meta_graph(r'/Users/zz/PycharmProjects/Mnist/Model/model.ckpt.meta')
    saver.restore(sess, r"/Users/zz/PycharmProjects/Mnist/Model/model.ckpt")
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y_ = graph.get_tensor_by_name("y_:0")

    with open(result_path, 'w') as result_file:
        for k in range(len(images)):
            picture = np.array(images[k]).reshape([1, 28, 28, 1])
            feed_dict = {x: picture, y_: label}
            tensor = graph.get_tensor_by_name(tensor_name)
            layer_output = sess.run(tensor, feed_dict)
            layer_output = np.array(layer_output).reshape([number_of_value])
            for value in layer_output:
                print(value, end='    ', file=result_file)
            print("\n", file=result_file)


# ------------------------------------------神经元输出----------------------------------------------------
# 测试集预测正确的图片的神经元输出
for i in range(10):
    Source_path = r'/Users/zz/PycharmProjects/Mnist/MNIST_data/training_data/class' + str(i) + \
                  '_correct_prediction.txt'

    # 第五层全连接层输出
    Result_path = r'/Users/zz/PycharmProjects/Mnist/MNIST_data/training_data/correct_neural_value/fc1/' + \
                  'class' + str(i) + '_correct_NeuralValue.txt'
    get_neural_value(source_path=Source_path, result_path=Result_path,
                     tensor_name="layer5-fc1/fc1:0", number_of_value=256)

    # 第六层全连接层输出
    Result_path = r'/Users/zz/PycharmProjects/Mnist/MNIST_data/training_data/correct_neural_value/fc2/' + \
                  'class' + str(i) + '_correct_NeuralValue.txt'
    get_neural_value(source_path=Source_path, result_path=Result_path,
                     tensor_name="layer6-fc2/fc2:0", number_of_value=128)


# 训练集预测正确的图片的神经元输出
for i in range(10):
    Source_path = r'/Users/zz/PycharmProjects/Mnist/MNIST_data/testing_data/class' + str(i) + \
                  '_correct_prediction.txt'

    # 第五层全连接层输出
    Result_path = r'/Users/zz/PycharmProjects/Mnist/MNIST_data/testing_data/correct_neural_value/fc1/' + \
                  'class' + str(i) + '_correct_NeuralValue.txt'
    get_neural_value(source_path=Source_path, result_path=Result_path,
                     tensor_name="layer5-fc1/fc1:0", number_of_value=256)

    # 第六层全连接层输出
    Result_path = r'/Users/zz/PycharmProjects/Mnist/MNIST_data/testing_data/correct_neural_value/fc2/' + \
                  'class' + str(i) + '_correct_NeuralValue.txt'
    get_neural_value(source_path=Source_path, result_path=Result_path,
                     tensor_name="layer6-fc2/fc2:0", number_of_value=128)


# 训练集原始类预测错误的图片的神经元输出
for i in range(10):
    Source_path = r'/Users/zz/PycharmProjects/Mnist/MNIST_data/testing_data/class' + str(i) + \
                  '_wrong_prediction.txt'

    # 第五层全连接层输出
    Result_path = r'/Users/zz/PycharmProjects/Mnist/MNIST_data/testing_data/wrong_neural_value/fc1/' + \
                  'class' + str(i) + '_wrong_NeuralValue.txt'
    get_neural_value(source_path=Source_path, result_path=Result_path,
                     tensor_name="layer5-fc1/fc1:0", number_of_value=256)

    # 第六层全连接层输出
    Result_path = r'/Users/zz/PycharmProjects/Mnist/MNIST_data/testing_data/wrong_neural_value/fc2/' + \
                  'class' + str(i) + '_wrong_NeuralValue.txt'
    get_neural_value(source_path=Source_path, result_path=Result_path,
                     tensor_name="layer6-fc2/fc2:0", number_of_value=128)


# 测试集预测正确的图片的神经元输出
for i in range(10):
    Source_path = r'/Users/zz/PycharmProjects/Mnist/MNIST_data/testing_data/wrong_prediction_to_' + str(i) + \
                  '.txt'

    # 第五层全连接层输出
    Result_path = r'/Users/zz/PycharmProjects/Mnist/MNIST_data/testing_data/wrong_to_neural_value/fc1/' + \
                  'wrong_to_' + str(i) + 'NeuralValue.txt'
    get_neural_value(source_path=Source_path, result_path=Result_path,
                     tensor_name="layer5-fc1/fc1:0", number_of_value=256)

    # 第六层全连接层输出
    Result_path = r'/Users/zz/PycharmProjects/Mnist/MNIST_data/testing_data/wrong_to_neural_value/fc2/' + \
                  'wrong_to_' + str(i) + 'NeuralValue.txt'
    get_neural_value(source_path=Source_path, result_path=Result_path,
                     tensor_name="layer6-fc2/fc2:0", number_of_value=128)
