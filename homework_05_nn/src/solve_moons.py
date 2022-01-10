import time
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

import sklearn.neural_network as NN
import Neural_Network as MyNN


def load_moons(size=200, rate=0.7, noise=0.20):
    # generate sample data
    np.random.seed(0)
    data, label = datasets.make_moons(size, noise=noise)
    # 随机打乱数据
    shuffled_index = np.random.permutation(size)
    data = data[shuffled_index]
    label = label[shuffled_index]
    # 分割数据集
    split_index = int(size * rate)
    train_data = data[:split_index]
    train_label = label[:split_index]
    test_data = data[split_index:]
    test_label = label[split_index:]
    return train_data, train_label, test_data, test_label


def plotFeature(data, label):
    plt.scatter(data[:, 0], data[:, 1], c=label, cmap=plt.cm.Spectral)
    plt.show()


if __name__ == "__main__":
    train_data, train_label, test_data, test_label = load_moons(size=1000, rate=0.75, noise=0.20)
    # =====================使用自己的方法做预测========================
    # 构建一个多层神经网络
    nn = MyNN.Neural_Network(nodes=[2, 8, 8, 2], rate=0.01, epoch=1000)

    # 开始训练并且计时
    start = time.clock()
    nn.train(train_data, train_label, show_process=False)
    end = time.clock()
    print("MyNN 训练用时:\t%f s" % (end - start))

    # 对训练集做预测并且输出准确度与结果
    train_result, _ = nn.predict(train_data)
    print("MyNN 对训练集做预测:\t", accuracy_score(train_label, train_result))
    plotFeature(train_data, train_result)

    # 对测试集做预测并且输出准确度与结果
    test_result, _ = nn.predict(test_data)
    print("MyNN 对测试集做预测:\t", accuracy_score(test_label, test_result))
    plotFeature(test_data, test_result)
    # =====================使用sklearn做预测========================
    # 构建一个多层神经网络
    # 因为这里使用的数据量很少，根据网上的介绍，使用lbfgs方法来优化权重，收敛更快效果也更好
    clf = NN.MLPClassifier(solver='lbfgs', activation='logistic', max_iter=300, alpha=1e-5, hidden_layer_sizes=(16, 16))
    # 开始训练并且计时
    start = time.clock()
    clf.fit(train_data, train_label)
    end = time.clock()
    print("sklearn 训练用时:\t%f s" % (end - start))

    # 对训练集做预测并且输出准确度与结果
    train_result = clf.predict(train_data)
    print("sklearn 对训练集做预测:\t", accuracy_score(train_label, train_result))
    plotFeature(train_data, train_result)

    # 对测试集做预测并且输出准确度与结果
    test_result = clf.predict(test_data)
    print("sklearn 对测试集做预测:\t", accuracy_score(test_label, test_result))
    plotFeature(test_data, test_result)
