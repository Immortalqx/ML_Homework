import time
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score

import sklearn.neural_network as NN
import Neural_Network as MyNN

fig_num = 0


def plotFeature(data, labels_):
    """
    二维空间显示聚类结果
    """
    # 获取簇集的个数
    clusterNum = len(set(labels_))
    # 内定的颜色种类
    scatterColors = ["orange", "purple", "cyan", "red", "green"]

    # 判断数据的维度是不是２
    if np.shape(data)[1] != 2:
        print("sorry,the dimension of your dataset is not 2!")
        return 1
    # 判断簇集类别是否大于５类
    if clusterNum > len(scatterColors):
        print("sorry,your k is too large,please add length of the scatterColors!")
        return 1

    # 散点图的绘制
    for i in range(clusterNum):
        colorSytle = scatterColors[i % len(scatterColors)]
        subCluster = data[np.where(labels_ == i)]
        plt.scatter(subCluster[:, 0], subCluster[:, 1], c=colorSytle, s=20)

    # 设置x、y轴和标题
    plt.xlabel("x-label")
    plt.ylabel("y-label")
    plt.show()


def load_circles(rate=0.7):
    # 从文本中加载数据
    dataset_circles = np.loadtxt("../dataset_circles.csv", delimiter=",")
    # 随机打乱数据
    data_size = np.shape(dataset_circles)[0]
    shuffled_index = np.random.permutation(data_size)
    dataset_circles = dataset_circles[shuffled_index]
    # 分割前两列为data
    data = dataset_circles.T[0:2]
    # 分割第三列为label
    label = dataset_circles.T[2:3]
    # 分割数据集
    split_index = int(data_size * rate)
    train_data = data.T[:split_index]
    train_label = label.T[:split_index]
    test_data = data.T[split_index:]
    test_label = label.T[split_index:]
    return train_data, train_label, test_data, test_label


if __name__ == "__main__":
    train_data, train_label, test_data, test_label = load_circles(rate=0.75)
    # =====================使用自己的方法做预测========================
    # 构建一个多层神经网络
    nn = MyNN.Neural_Network(nodes=[2, 8, 8, 2], rate=0.01, epoch=2000)

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
    clf = NN.MLPClassifier(solver='lbfgs', activation='logistic', max_iter=300, alpha=1e-5, hidden_layer_sizes=(30, 30))
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
