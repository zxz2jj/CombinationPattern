import tensorflow as tf
import numpy as np

f_test_0 = open(r"/Users/zz/wrong_prediction_to_0.txt", 'w')
f_test_1 = open(r"/Users/zz/wrong_prediction_to_1.txt", 'w')
f_test_2 = open(r"/Users/zz/wrong_prediction_to_2.txt", 'w')
f_test_3 = open(r"/Users/zz/wrong_prediction_to_3.txt", 'w')
f_test_4 = open(r"/Users/zz/wrong_prediction_to_4.txt", 'w')
f_test_5 = open(r"/Users/zz/wrong_prediction_to_5.txt", 'w')
f_test_6 = open(r"/Users/zz/wrong_prediction_to_6.txt", 'w')
f_test_7 = open(r"/Users/zz/wrong_prediction_to_7.txt", 'w')
f_test_8 = open(r"/Users/zz/wrong_prediction_to_8.txt", 'w')
f_test_9 = open(r"/Users/zz/wrong_prediction_to_9.txt", 'w')


def classify_data_by_classes(
        img, label_y, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9):
    """
    将一个数据集按照类别，分别输出到不同的文件中
    """
    _f = 0
    if label_y == 0:
        _f = f0
    elif label_y == 1:
        _f = f1
    elif label_y == 2:
        _f = f2
    elif label_y == 3:
        _f = f3
    elif label_y == 4:
        _f = f4
    elif label_y == 5:
        _f = f5
    elif label_y == 6:
        _f = f6
    elif label_y == 7:
        _f = f7
    elif label_y == 8:
        _f = f8
    elif label_y == 9:
        _f = f9
    image = np.array(img).reshape(28, 28)
    for row in range(28):
        for col in range(28):
            print(image[row][col], end="    ", file=_f)
    print("\n", file=_f)


# load_model_and_predict
for target in range(10):
    print("当前分类到第", target, "类\n")
    with open(r"/Users/zz/PycharmProjects/fashionMnist/MNIST_data/testing_data/test_images_class_" +
              str(target)+".txt", 'r') as f:
        data = f.read()
        images = []
        numlist = data.split()
        for number_str in numlist:
            number_float = float(number_str)
            images.append(number_float)
    images = np.array(images).reshape([-1, 784])
    label = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    sess = tf.Session()
    saver = tf.train.import_meta_graph(r'/Users/zz/PycharmProjects/fashionMnist/Model/model.ckpt.meta')
    saver.restore(sess, r"/Users/zz/PycharmProjects/fashionMnist/Model/model.ckpt")
    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y_ = graph.get_tensor_by_name("y_:0")
    # keep_prob = graph.get_tensor_by_name("keep_prob:0")

    # f_c = open(r'/Users/zz/PycharmProjects/fashionMnist/MNIST_data/testing_data/class' +
    #            str(target)+'_correct_prediction.txt', 'w')
    # f_w = open(r'/Users/zz/PycharmProjects/fashionMnist/MNIST_data/testing_data/class' +
    #            str(target)+'_wrong_prediction.txt', 'w')
    f_c = open(r'/Users/zz/class' + str(target)+'_correct_prediction.txt', 'w')
    f_w = open(r'/Users/zz/class' + str(target)+'_wrong_prediction.txt', 'w')

    wrong_prediction = []
    wrong_picture = []
    correct_picture = []
    for i in range(len(images)):
        picture = np.array(images[i]).reshape([1, 28, 28, 1])
        feed_dict = {x: picture, y_: label}
        y = graph.get_tensor_by_name("layer7-fc3/output:0")
        yy = sess.run(y, feed_dict)
        classes = sess.run(tf.argmax(yy, 1))
        if classes != target:
            classify_data_by_classes(picture, classes, f_test_0, f_test_1, f_test_2, f_test_3, f_test_4,
                                     f_test_5, f_test_6, f_test_7, f_test_8, f_test_9)
            wrong_prediction.append(classes)
            # wrong_picture.append(np.reshape(picture, [784]))
            for k in range(28):
                for s in range(28):
                    print(picture[0][k][s][0], end="    ", file=f_w)
            print("\n", file=f_w)
        else:
            # correct_picture.append(np.reshape(picture, [784]))
            for k in range(28):
                for s in range(28):
                    print(picture[0][k][s][0], end="    ", file=f_c)
            print("\n", file=f_c)
    print("第", target, "类的错误预测样本有：", len(wrong_prediction), "\n", file=f_w)
    print(wrong_prediction, file=f_w)
    # print(correct_picture, file=f_c)
    # print(wrong_picture, file=f_w)
    f_c.close()
    f_w.close()














# f_test_0 = open(r"/Users/zz/PycharmProjects/fashionMnist/MNIST_data/testing_data/wrong_prediction_to_0.txt", 'w')
# f_test_1 = open(r"/Users/zz/PycharmProjects/fashionMnist/MNIST_data/testing_data/wrong_prediction_to_1.txt", 'w')
# f_test_2 = open(r"/Users/zz/PycharmProjects/fashionMnist/MNIST_data/testing_data/wrong_prediction_to_2.txt", 'w')
# f_test_3 = open(r"/Users/zz/PycharmProjects/fashionMnist/MNIST_data/testing_data/wrong_prediction_to_3.txt", 'w')
# f_test_4 = open(r"/Users/zz/PycharmProjects/fashionMnist/MNIST_data/testing_data/wrong_prediction_to_4.txt", 'w')
# f_test_5 = open(r"/Users/zz/PycharmProjects/fashionMnist/MNIST_data/testing_data/wrong_prediction_to_5.txt", 'w')
# f_test_6 = open(r"/Users/zz/PycharmProjects/fashionMnist/MNIST_data/testing_data/wrong_prediction_to_6.txt", 'w')
# f_test_7 = open(r"/Users/zz/PycharmProjects/fashionMnist/MNIST_data/testing_data/wrong_prediction_to_7.txt", 'w')
# f_test_8 = open(r"/Users/zz/PycharmProjects/fashionMnist/MNIST_data/testing_data/wrong_prediction_to_8.txt", 'w')
# f_test_9 = open(r"/Users/zz/PycharmProjects/fashionMnist/MNIST_data/testing_data/wrong_prediction_to_9.txt", 'w')
#
#
# def classify_data_by_classes(
#         img, label_y, f0, f1, f2, f3, f4, f5, f6, f7, f8, f9):
#     """
#     将一个数据集按照类别，分别输出到不同的文件中
#     """
#     _f = 0
#     if label_y == 0:
#         _f = f0
#     elif label_y == 1:
#         _f = f1
#     elif label_y == 2:
#         _f = f2
#     elif label_y == 3:
#         _f = f3
#     elif label_y == 4:
#         _f = f4
#     elif label_y == 5:
#         _f = f5
#     elif label_y == 6:
#         _f = f6
#     elif label_y == 7:
#         _f = f7
#     elif label_y == 8:
#         _f = f8
#     elif label_y == 9:
#         _f = f9
#     image = np.array(img).reshape(28, 28)
#     for row in range(28):
#         for col in range(28):
#             print(image[row][col], end="    ", file=_f)
#     print("\n", file=_f)
#
#
# # 将测试集预测错误的图片，结果为i类的图片写在wrong_prediction_to_i.txt
# for target in range(0, 10):
#     print("当前分类到第", target, "类\n")
#     with open(r"/Users/zz/PycharmProjects/fashionMnist/MNIST_data/testing_data/test_images_class_" +
#               str(target)+".txt", 'r') as f:
#         data = f.read()
#         images = []
#         numlist = data.split()
#         for number_str in numlist:
#             number_float = float(number_str)
#             images.append(number_float)
#     images = np.array(images).reshape([-1, 784])
#     label = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
#
#     sess = tf.Session()
#     saver = tf.train.import_meta_graph(r'/Users/zz/PycharmProjects/fashionMnist/Model/model.ckpt.meta')
#     saver.restore(sess, r"/Users/zz/PycharmProjects/fashionMnist/Model/model.ckpt")
#     graph = tf.get_default_graph()
#     x = graph.get_tensor_by_name("x:0")
#     y_ = graph.get_tensor_by_name("y_:0")
#     # keep_prob = graph.get_tensor_by_name("keep_prob:0")
#
#     for i in range(len(images)):
#         picture = np.array(images[i]).reshape([1, 28, 28, 1])
#         feed_dict = {x: picture, y_: label}
#         y = graph.get_tensor_by_name("layer7-fc3/output:0")
#         yy = sess.run(y, feed_dict)
#         classes = sess.run(tf.argmax(yy, 1))
#         if classes != target:
#             classify_data_by_classes(picture, classes, f_test_0, f_test_1, f_test_2, f_test_3, f_test_4,
#                                      f_test_5, f_test_6, f_test_7, f_test_8, f_test_9)
#
