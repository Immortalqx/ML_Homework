import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
import time


def compute_squared_EDM(X):
    """
    计算各个点之间的距离，返回一个向量
    """
    # 先使用pdist函数计算出一个N×N的对称矩阵
    # 再使用squareform转化为向量输出，具体原理是对右上三角逐行扫描保存
    return squareform(pdist(X, metric="euclidean"))


def DBSCAN(data, eps, minPts):
    """
    DBSCAN算法核心过程
    主要参考: https://zhuanlan.zhihu.com/p/77043965
    """
    # 获得距离矩阵
    disMat = compute_squared_EDM(data)
    # 获得数据的行和列(一共有n条数据)
    n, m = data.shape
    # 将矩阵的中小于minPts的数赋予1，大于minPts的数赋予零，然后1代表对每一行求和,然后求核心点坐标的索引
    core_points_index = np.where(np.sum(np.where(disMat <= eps, 1, 0), axis=1) >= minPts)[0]
    # 初始化类别，-1代表未分类。
    labels = np.full((n,), -1)
    clusterId = 0
    # 遍历所有的核心点
    for pointId in core_points_index:
        # 如果核心点未被分类，将其作为的种子点，开始寻找相应簇集
        if labels[pointId] == -1:
            # 首先将点pointId标记为当前类别(即标识为已操作)
            labels[pointId] = clusterId
            # 然后寻找种子点的eps邻域且没有被分类的点，将其放入种子集合
            neighbour = np.where((disMat[:, pointId] <= eps) & (labels == -1))[0]
            seeds = set(neighbour)
            # 通过种子点，开始生长，寻找密度可达的数据点，一直到种子集合为空，一个簇集寻找完毕
            while len(seeds) > 0:
                # 弹出一个新种子点
                newPoint = seeds.pop()
                # 将newPoint标记为当前类
                labels[newPoint] = clusterId
                # 寻找newPoint种子点eps邻域（包含自己）
                queryResults = np.where(disMat[:, newPoint] <= eps)[0]
                # 如果newPoint属于核心点，那么newPoint是可以扩展的，即密度是可以通过newPoint继续密度可达的
                if len(queryResults) >= minPts:
                    # 将邻域内且没有被分类的点压入种子集合
                    for resultPoint in queryResults:
                        if labels[resultPoint] == -1:
                            seeds.add(resultPoint)
            # 簇集生长完毕，寻找到一个类别
            clusterId = clusterId + 1
    return labels


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
    plt.title("DBSCAN cluster result")

    plt.show()


if __name__ == "__main__":
    # 从文本中加载数据
    data = np.loadtxt("../dataset_circles.csv", delimiter=",")
    # 分割前两列作为训练集
    train_data = data.T[0:2]
    # 开始计时
    start = time.clock()
    # DBSCAN算法
    labels = DBSCAN(train_data.T, 6, 8)
    # 结束计时
    end = time.clock()
    # 分割第三列作为测试集
    test_data = data.T[2:3]
    # 评估聚类算法的结果
    evaluate(test_data.T, labels)
    # 输出聚类算法所用时间
    print("time = %s" % str(end - start))
    # 绘图显示
    plotFeature(train_data.T, labels)
