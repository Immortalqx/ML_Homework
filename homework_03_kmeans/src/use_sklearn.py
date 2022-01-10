import time
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

# TODO 数据的维度不太清楚，使用ARI时一开始出了问题
#  然后不使用聚类变换用KMeans得到的结果正确性大概是50%，但ARI计算出来的结果是-0.25%？

# 从文本中加载数据
data = np.loadtxt("../dataset_circles.csv", delimiter=",")
# 分割前两列作为训练集
train_data = data.T[0:2]
# 进行特征变化，这里很容易想到将特征变化为点到原点的距离（这里用的平方）
trans_data = np.array([train_data[0] ** 2 + train_data[1] ** 2])
# 分割第三列作为测试集
test_data = data.T[2:3]
# 图片数目
figure = 0


def MY_DBSCAN():
    # 构造一个ϵ=6,MinPts=8的聚类器，距离使用欧式距离
    estimator = DBSCAN(eps=6, min_samples=8, metric='euclidean')
    # 开始计时
    start = time.clock()
    # 聚类数据
    estimator.fit(train_data.T)
    # 结束计时
    end = time.clock()
    # 输出聚类算法的正确率和所用时间
    print("DBSCAN:")
    evaluate(test_data.T, estimator.labels_)
    new_evaluate(test_data[0].T, estimator.labels_)
    print("time = %s" % str(end - start))
    # 绘制散点图
    plotFeature(train_data.T, estimator.labels_, "DBSCAN")


def MY_KMEANS():
    # 构造一个K=2的聚类器
    estimator = KMeans(n_clusters=2, random_state=9)
    # 开始计时
    start = time.clock()
    # 聚类数据
    # estimator.fit(train_data.T)
    estimator.fit(trans_data.T)
    # 结束计时
    end = time.clock()
    # 输出聚类算法的正确率和所用时间
    print("K-means:")
    evaluate(test_data.T, estimator.labels_)
    new_evaluate(test_data[0].T, estimator.labels_)
    print("time = %s" % str(end - start))
    # 绘制散点图
    plotFeature(train_data.T, estimator.labels_, "K-means")


# 这个是之前手写的，因为只有两个类，所以要么对应上，要么对应不上，取score和(1-score)中值更大的即可
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
    if 1.0 - score > score:
        score = 1.0 - score
    print("score = %f" % (100.0 * score))


def new_evaluate(data, labels):
    """
    使用ARI评估聚类性能，data必须是1D的
    """
    # ARI取值范围为[-1,1]，值越大越好，反映两种划分的重叠程度，使用该度量指标需要数据本身有类别标记。
    score = adjusted_rand_score(data, labels)
    print("score = %f" % (100.0 * score))


def plotFeature(data, labels_, title):
    """
    二维空间显示聚类结果
    """
    global figure
    figure += 1
    plt.figure(figure)
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
    plt.title(title + " cluster result")


if __name__ == "__main__":
    MY_DBSCAN()
    MY_KMEANS()
    plt.show()
