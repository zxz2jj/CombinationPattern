from tensorflow.examples.tutorials.mnist import input_data
import os
import numpy as np
import tensorflow as tf

MODEL_SAVE_PATH = r"/Users/zz/PycharmProjects/fashionMnist/Model"
MODEL_NAME = "model.ckpt"


def load_data():
    # reshape mnist数据集中训练数据和测试数据
    mnist = input_data.read_data_sets(r"/Users/zz/PycharmProjects/fashionMnist/MNIST_data", one_hot=True)
    train_data, train_label = mnist.train.images, mnist.train.labels
    train_data = np.reshape(train_data, (train_data.shape[0], 28, 28, 1))
    test_data, test_label = mnist.test.images, mnist.test.labels
    test_data = np.reshape(test_data, (test_data.shape[0], 28, 28, 1))

    # 对训练数据做随机打乱
    train_image_num = len(train_data)
    train_image_index = np.arange(train_image_num)
    np.random.shuffle(train_image_index)
    train_data = train_data[train_image_index]
    train_label = train_label[train_image_index]
    return test_data, test_label, train_data, train_label


# 搭建CNN
x = tf.placeholder(tf.float32, [None, 28, 28, 1], name='x')
y_ = tf.placeholder(tf.float32, [None, 10], name='y_')


def inference(input_tensor, train, regularizer):
    # 第一层：卷积层，过滤器的尺寸为5×5，深度为6,使用全0补充，步长为1。
    # 尺寸变化：28×28×1->28×28×32
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.Variable(initial_value=tf.truncated_normal(shape=[5, 5, 1, 32], stddev=0.1), name='weight')
        conv1_biases = tf.Variable(initial_value=tf.constant(shape=[32], value=0.0), name='bias')
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        layer1 = tf.nn.relu(tf.add(conv1, conv1_biases), name="conv1")

    # 第二层：池化层，过滤器的尺寸为2×2，使用全0补充，步长为2。
    # 尺寸变化：28×28×32->14×14×32
    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(layer1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool1")

    # 第三层：卷积层，过滤器的尺寸为5×5，深度为16,使用全0补充，步长为1。
    # 尺寸变化：14×14×32->14×14×64
    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.Variable(initial_value=tf.truncated_normal(shape=[5, 5, 32, 64], stddev=0.1), name='weight')
        conv2_biases = tf.Variable(initial_value=tf.constant(shape=[64], value=0.0), name='bias')
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        layer3 = tf.nn.relu(tf.add(conv2, conv2_biases), name="conv2")

    # 第四层：池化层，过滤器的尺寸为2×2，使用全0补充，步长为2。
    # 尺寸变化：14×14×64->7×7×64
    with tf.variable_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(layer3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name="pool2")

    # 第五层：全连接层，7×7×64->256
    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.Variable(initial_value=tf.truncated_normal(shape=[7*7*64, 256], stddev=0.1), name='weight')
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.Variable(initial_value=tf.constant(shape=[256], value=0.1), name='bias')
        flat_pool2 = tf.reshape(pool2, shape=[-1, 7*7*64])
        fc1 = tf.nn.relu(tf.matmul(flat_pool2, fc1_weights) + fc1_biases, name="fc1")
        tf.add_to_collection("fc1", fc1)
        if train:
            fc1 = tf.nn.dropout(fc1, 0.5)

    # 第六层：全连接层，256->128的全连接
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.Variable(initial_value=tf.truncated_normal(shape=[256, 128], stddev=0.1), name='weight')
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.Variable(initial_value=tf.constant(shape=[128], value=0.1), name='bias')
        fc2 = tf.nn.relu(tf.matmul(fc1, fc2_weights) + fc2_biases, name="fc2")
        tf.add_to_collection("fc2", fc2)
        if train:
            fc2 = tf.nn.dropout(fc2, 0.5)

    # 第六层：全连接层，128->10的全连接
    with tf.variable_scope('layer7-fc3'):
        fc3_weights = tf.Variable(initial_value=tf.truncated_normal(shape=[128, 10], stddev=0.1), name='weight')
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc3_weights))
        fc3_biases = tf.Variable(initial_value=tf.constant(shape=[10], value=0.1), name='bias')
        logit = tf.add(tf.matmul(fc2, fc3_weights), fc3_biases, name="output")
        tf.add_to_collection("fc3", logit)
    return logit


def main(test_data, test_label, train_data, train_label, command=None):
    if command is None:
        print("Please input train or test!")
        return
    elif command == "test":
        y = inference(x, False, None)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

        sess = tf.Session()
        saver = tf.train.Saver()
        saver.restore(sess, r"/Users/zz/PycharmProjects/fashionMnist/Model/model.ckpt")
        graph = tf.get_default_graph()
        input_x = graph.get_tensor_by_name("x:0")
        label_y_ = graph.get_tensor_by_name("y_:0")
        acc = sess.run(accuracy, feed_dict={input_x: test_data, label_y_: test_label})
        print("testing accuracy:", acc)
        return

    # 每次获取batch_size个样本进行训练或测
    elif command == "train":
        regularizer = tf.contrib.layers.l2_regularizer(0.001)
        y = inference(x, True, regularizer)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses'))
        train_op = tf.train.AdamOptimizer(0.001).minimize(loss)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")

        saver = tf.train.Saver()
        # 可以在这里添加加载模型继续训练相关代码
        # 创建Session会话
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            def get_batch(data, label, batch_size):
                for start_index in range(0, len(data) - batch_size + 1, batch_size):
                    slice_index = slice(start_index, start_index + batch_size)
                    yield data[slice_index], label[slice_index]

            # 将所有样本训练10次，每次训练中以64个为一组训练完所有样本。
            # train_num可以设置大一些。
            train_num = 100
            batchsize = 64
            for i in range(train_num):
                train_loss, train_acc, batch_num = 0, 0, 0
                for train_data_batch, train_label_batch in get_batch(train_data, train_label, batchsize):
                    _, err, acc = sess.run([train_op, loss, accuracy], feed_dict={x: train_data_batch, y_: train_label_batch})
                    train_loss += err
                    train_acc += acc
                    batch_num += 1
                print("epoch:", i)
                print("train loss:", train_loss/batch_num)
                print("train acc:", train_acc/batch_num)
            saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME))
    else:
        return


if __name__ == "__main__":
    ted, tel, trd, trl = load_data()
    main(ted, tel, trd, trl, "test")
