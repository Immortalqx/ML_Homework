import time
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


def compute_distances(A, B):
    """
    计算两组向量之间的距离，两个向量要有一样的维度和长度
    """
    # 与sqrt(sum(power(vecA - vecB, 2)))相比，这个可以直接计算一整组的距离，更方便
    return cdist(A, B, metric="euclidean")


def K_means(data, K, N):
    """
    KMeans聚类算法的具体实现
    :param data:
    :param K: 聚类的个数
    :param N: 算法的迭代次数
    :return: 类别标志labels与中心坐标centerPoints
    """
    # 获取数据行数
    n = np.shape(data)[0]
    # 从n条数据中随机选择K条，作为初始中心向量
    # centerId是初始中心向量的索引坐标
    centerId = random.sample(range(0, n), K)
    # 获得初始中心向量，共k个
    centerPoints = data[centerId]
    # 计算data到centerPoints的距离矩阵
    # dist[i][:],是i个点到各个中心点的距离
    dist = compute_distances(data, centerPoints)
    # 这里进行第一次判断，距离哪个中心点进就判断为哪一类
    # axis=1寻找每一行中最小值的索引
    labels = np.argmin(dist, axis=1).squeeze()
    # 迭代次数
    count = 0
    # 循环次数小于迭代次数，一直迭代
    while count < N:
        for i in range(K):
            # 重新计算每一个类别的中心点
            centerPoints[i] = np.mean(data[np.where(labels == i)], 0)
        # 重新计算距离矩阵
        dist = compute_distances(data, centerPoints)
        # 重新分类
        labels = np.argmin(dist, axis=1).squeeze()
        # 迭代次数加1
        count += 1
    # 返回类别标识，中心坐标
    return labels, centerPoints


def evaluate(data, labels):
    """
    评估聚类的结果
    """
    num, div = np.shape(data)
    # 判断数据的维度是不是1
    if div != 1:
        print("sorry,the dimension of your dataset is not 1!")
        return 1

    count = 0
    for i in range(num):
        if data[i] == labels[i]:
            count += 1
        i += 1
    score = count * 1.0 / num
    # 因为分类是随机的，不一定对应的上（比如0和1对应，1和0对应，导致正确率为0），所以要调整一下
    if 1 - score > score:
        score = 1 - score
    print("score = %f" % (100.0 * score))


def getCenterPoint(data, K, labels):
    centerPoints = data[:K]
    for i in range(K):
        # 重新计算每一个类别的中心点
        centerPoints[i] = np.mean(data[np.where(labels == i)], 0)
    return centerPoints


def plotFeature(data, labels_, centerPoints):
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

    # 中心点的绘制
    mark = ["*", "^", "o", "X", "D"]  # 这个是设置绘制的形状
    label = ["0", "1", "2", "3", "4"]  # label的作用是设置图例
    c = ["blue", "red", "peru", "violet", "black"]
    for i in range(clusterNum):
        plt.plot(centerPoints[i, 0], centerPoints[i, 1],
                 mark[i], markersize=15, label=label[i], c=c[i])
        plt.legend(loc="upper left")  # 图例

    # 设置x、y轴和标题
    plt.xlabel("x-label")
    plt.ylabel("y-label")
    plt.title("k-means cluster result")

    plt.show()


if __name__ == "__main__":
    # 用户定义聚类数
    K = 2
    # 从文本中加载数据
    data = np.loadtxt("../dataset_circles.csv", delimiter=",")
    # 分割前两列作为训练集
    train_data = data.T[0:2]
    # 进行特征变化，这里很容易想到将特征变化为点到原点的距离（这里用的平方）
    trans_data = np.array([train_data[0] ** 2 + train_data[1] ** 2])
    # 分割第三列作为测试集
    test_data = data.T[2:3]
    # 开始计时
    start = time.clock()
    # 执行KMeans算法
    labels, _ = K_means(trans_data.T, K, 30)
    # 结束计时
    end = time.clock()
    # 由于这里进行特征变化了，返回的中心点是一维的，因此要根据聚类结果重新计算中心点
    centerPoints = getCenterPoint(train_data.T, K, labels)
    # 评估聚类算法的结果
    evaluate(test_data.T, labels)
    # 输出聚类算法所用时间
    print("time = %s" % str(end - start))
    # 绘图显示
    plotFeature(train_data.T, labels, centerPoints)
