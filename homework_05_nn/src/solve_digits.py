import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.metrics import confusion_matrix, accuracy_score

import sklearn.neural_network as NN
import Neural_Network as MyNN

fig_num = 0


def display(images, ground_truth, predict, rate, fig_name):
    """
    画出预测结果
    :param images: 数字的图片
    :param ground_truth: 真值
    :param predict: 预测值
    :param fig_name: 图片名字
    """
    # plot the digits
    global fig_num
    fig_num += 1
    if fig_name is None:
        fig = plt.figure(fig_num, figsize=(6, 6))  # figure size in inches
    else:
        fig = plt.figure(fig_name, figsize=(6, 6))  # figure size in inches
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    size = np.shape(predict)[0]
    if size > 56:
        size = 56
    # plot the digits: each image is 8x8 pixels
    for i in range(size):
        ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
        ax.imshow(images[i], cmap=plt.cm.binary)
        # label the image with the target value
        ax.text(0, 7, str(ground_truth[i]))
        if int(ground_truth[i]) == int(predict[i]):
            ax.text(6, 7, str(int(predict[i])), color='green', size=15)
            if rate is not None:
                ax.text(4, 1, str(int(rate[i] * 100) / 100), color='green', size=10)
        else:
            ax.text(6, 7, str(int(predict[i])), color='red', size=20)
            if rate is not None:
                ax.text(4, 1, str(int(rate[i] * 100) / 100), color='red', size=10)

    # plot confusion matrix
    cm = confusion_matrix(ground_truth, predict)
    plt.matshow(cm)
    plt.colorbar()
    plt.title(fig_name + ' ' + 'Confusion Matrix')
    plt.ylabel('Groundtruth')
    plt.xlabel('Predict')


def load_data(train_size=0.75):
    """
    加载数据同时分割出训练集和测试集
    :param train_size: 训练集的比例
    :return 分割好的数据集
    """
    # 加载数据
    # 这个函数返回的应该是一个类实例，里面有很多的变量，需要“单独的拿出来用”
    digits = load_digits()
    # 输出查看一下这个类的属性和方法
    # print(dir(digits))
    # 获得数据的大小
    data_size = np.shape(digits.data)[0]
    # 将数据打乱，这里需要把这个类的要用上的变量都对应上，有点麻烦（是否有更简便的方法？）
    shuffled_index = np.random.permutation(data_size)

    digits.data = digits.data[shuffled_index]
    digits.images = digits.images[shuffled_index]
    digits.target = digits.target[shuffled_index]

    # 分割数据集
    split_index = int(data_size * train_size)

    train_data = digits.data[0:split_index]
    train_target = digits.target[0:split_index]
    train_images = digits.images[0:split_index]

    test_data = digits.data[split_index:]
    test_target = digits.target[split_index:]
    test_images = digits.images[split_index:]

    return train_data, train_target, train_images, test_data, test_target, test_images


if __name__ == "__main__":
    # FIXME:
    #  1. 数据的加载应该没问题，在逻辑回归里面得到的结果是正确的
    #  但放到神经网络里面，如果给的训练集过多（1000以上），就没有收敛，但给的数据集少一些效果好一些
    #  2. 感觉是有地方溢出了，导致直接就没有收敛，猜测是sigmoid函数溢出
    #  3. 测试发现sigmoid函数没有溢出，可能是学习率过低，陷入了局部最优解，但调大学习率就会报警，说sigmoid函数溢出
    #  4. 调整sigmoid函数后，调大学习率没有影响，但把学习率调小会得到一个很好的结果。
    #  问题：如何获得最优的学习率？如何知道学习率是过大还是过小？
    # 加载并且分割各种数据
    train_data, train_target, train_images, test_data, test_target, test_images = load_data(train_size=0.75)
    # =====================使用自己的方法做预测========================
    nn = MyNN.Neural_Network(nodes=[64, 32, 32, 10], rate=0.001, epoch=1000)

    # 开始训练并且计时
    # FIXME 这里时间测量的不对，但其他地方没有问题
    start = time.clock()
    nn.train(train_data, train_target, show_process=True)
    end = time.clock()
    print("MyNN 训练用时:\t%f s" % (end - start))

    # 对训练集做预测并且输出准确度与结果
    train_result, train_rate = nn.predict(train_data)
    print("MyNN 对训练集做预测:\t", accuracy_score(train_target, train_result))
    display(train_images, train_target, train_result, train_rate, "My Train")

    # 对测试集做预测并且输出准确度与结果
    test_result, test_rate = nn.predict(test_data)
    print("MyNN 对测试集做预测:\t", accuracy_score(test_target, test_result))
    display(test_images, test_target, test_result, test_rate, "My Test")
    # =====================使用sklearn做预测========================
    # 构建一个多层神经网络
    # 因为这里使用的数据量很少，根据网上的介绍，使用lbfgs方法来优化权重，收敛更快效果也更好
    clf = NN.MLPClassifier(solver='lbfgs', activation='logistic', max_iter=300, alpha=1e-5, hidden_layer_sizes=(64, 64))
    # 开始训练并且计时
    start = time.clock()
    clf.fit(train_data, train_target)
    end = time.clock()
    print("sklearn 训练用时:\t%f s" % (end - start))

    # 对训练集做预测并且输出准确度与结果
    train_result = clf.predict(train_data)
    print("sklearn 对训练集做预测:\t", accuracy_score(train_target, train_result))
    display(train_images, train_target, train_result, None, "sklearn Train")

    # 对测试集做预测并且输出准确度与结果
    test_result = clf.predict(test_data)
    print("sklearn 对测试集做预测:\t", accuracy_score(test_target, test_result))
    display(test_images, test_target, test_result, None, "sklearn Test")

    plt.show()
