import numpy as np
from sklearn.metrics import accuracy_score


# TODO
#  要求
#  1. 先用函数的方式实现网络的正向计算和反向误差传播，权值更新。【done】
#  2. 构思并实现基于类的神经网络程序。【done】
#  3. 构建多分类的网络（可以使用`dataset_digits`）【done】
#  4. 学习`softmax`和`cross entropy`的方法，并实现类别所属概率的输出。【done】
#  5. 对比自己实现与sklearn的方法的精度。【done】
#  6. 如何将分类错误的样本可视化出来？【done】
#  思路
#  1. 先使用sigmoid函数把多分类神经网络简单搭建出来，搞清楚整个的流程【done】
#  2. 完成可视化的部分，把图和分类错误的样本画出来【done】
#  3. 与sklearn对比【done】
#  4. 搞懂softmax函数和交叉熵损失在神经网络中的应用【done】
#  5. 输出层使用softmax函数 【done】
#  6. 反向传播尝试一下交叉熵函数 【done】（反向传播用的就是交叉熵函数）
#  7. 加入正则化等，防止输入数据过多导致过拟合的情况 【quit】（过拟合是训练集效果好，测试集不行，但这里的情况显然不是，就不做了）
#  8. 尝试一下sklearn的方法，自动判断输入层和输出层，从而只需要指定隐藏层 【quit】

# def sigmoid(x):
#     """
#     sigmoid函数
#     """
#     return 1.0 / (1 + np.exp(-x))

def sigmoid(Z):
    """
    sigmoid函数，解决了溢出的问题
    把大于0和小于0的元素分别处理
    原来的sigmoid函数是 1/(1+np.exp(-Z))
    当Z是比较小的负数时会出现上溢，此时可以通过计算exp(Z) / (1+exp(Z)) 来解决
    """
    mask = (Z > 0)
    positive_out = np.zeros_like(Z, dtype='float64')
    negative_out = np.zeros_like(Z, dtype='float64')

    # 大于0的情况
    positive_out = 1 / (1 + np.exp(-Z, positive_out, where=mask))
    # 清除对小于等于0元素的影响
    positive_out[~mask] = 0

    # 小于等于0的情况
    expZ = np.exp(Z, negative_out, where=~mask)
    negative_out = expZ / (1 + expZ)
    # 清除对大于0元素的影响
    negative_out[mask] = 0

    return positive_out + negative_out


# FIXME softmax函数输出的概率比较低，比如错误分类概率为0.18，而正确概率为0.22，相差不太大
def softmax(x):
    """
    对输入x的每一行计算softmax。
    该函数对于输入是向量（将向量视为单独的行）或者矩阵（M x N）均适用。
    代码利用softmax函数的性质: softmax(x) = softmax(x + c)
    :param x: 一个N维向量，或者M x N维numpy矩阵.
    """
    # 根据输入类型是矩阵还是向量分别计算softmax
    if len(x.shape) > 1:
        # 矩阵
        tmp = np.max(x, axis=1)  # 得到每行的最大值，用于缩放每行的元素，避免溢出。 shape为(x.shape[0],)
        x -= tmp.reshape((x.shape[0], 1))  # 利用性质缩放元素
        x = np.exp(x)  # 计算所有值的指数
        tmp = np.sum(x, axis=1)  # 每行求和
        x /= tmp.reshape((x.shape[0], 1))  # 求softmax
    else:
        # 向量
        tmp = np.max(x)  # 得到最大值
        x -= tmp  # 利用最大值缩放数据
        x = np.exp(x)  # 对所有元素求指数
        tmp = np.sum(x)  # 求元素和
        x /= tmp  # 求somftmax
    return x


class Neural_Network:
    """
    多层全连接神经网络
    """

    def __init__(self, nodes, rate=0.01, epoch=1000):
        """
        初始化一个神经网络实例
        :param nodes: 网络层次结构
        :param rate: 学习速率
        :param epoch: 最大迭代次数
        """
        # 网络层次结构
        self.nodes = nodes
        # 学习速率
        self.rate = rate
        # 最大迭代次数
        self.epoch = epoch

        # 偏置矩阵
        self.B = []
        # 权重矩阵
        self.W = []
        # 每一层的输出
        self.Z = []

        # 初始化权重与偏重
        for i in range(len(self.nodes) - 1):  # 对于每一层网络，根据该层节点数目进行初始化
            # 权重矩阵，不能初始化为0或1，不然迭代会失去梯度！
            w = np.random.randn(self.nodes[i], self.nodes[i + 1]) / np.sqrt(self.nodes[i])
            b = np.ones((1, self.nodes[i + 1]))
            self.W.append(w)
            self.B.append(b)

    def forward(self, data):
        """
        正向传播函数
        """
        # 存放每层输出
        Z = []
        x = data
        for j in range(len(self.nodes) - 1):
            x = sigmoid(x.dot(self.W[j]) + self.B[j])
            # 所有层输出存入list中
            Z.append(x)
        self.Z = Z

    def backpropagation(self, data, t):
        """
        反向传播函数
        """
        D = []
        d = t
        # 计算delta
        n_layer = len(self.nodes)
        for j in range(n_layer - 1, 0, -1):
            if j == n_layer - 1:  # 如果是输出层
                # delta = y * (1 - y) * (t - y)
                d = self.Z[-1] * (1 - self.Z[-1]) * (t - self.Z[-1])
            else:  # 如果是隐藏层
                # delta = y * (1-y) * sum( delta_k * w_k)
                d = self.Z[j - 1] * (1 - self.Z[j - 1]) * np.dot(d, self.W[j].T)
            # 存入list中，反向计算，后计算的存前面
            D.insert(0, d)

        # 更新权重和偏置
        # w += n * delta
        self.W[0] += self.rate * np.dot(data.T, D[0])
        self.B[0] += self.rate * np.sum(D[0], axis=0)
        for k in range(1, len(self.nodes) - 1):
            self.W[k] += self.rate * np.dot(self.Z[k - 1].T, D[k])
            self.B[k] += self.rate * np.sum(D[k], axis=0)

    def train(self, data, label, show_process=False):
        """
        进行训练
        :param data: 训练集数据
        :param label: 训练集标签
        :param show_process: 是否展示迭代过程
        """
        # 根据样本个数创建正确结果矩阵，每个样本对应结果矩阵中正确的结果位置值1，其他置0
        t = np.zeros((np.shape(data)[0], self.nodes[-1]))
        for i in range(self.nodes[-1]):
            t[np.where(label == i), i] = 1

        for i in range(self.epoch):
            self.forward(data)

            if show_process:
                loss = np.sum((t - self.Z[-1]) ** 2)
                predict = np.argmax(self.Z[-1], axis=1)
                accuracy = accuracy_score(label, predict)
                print("Loss:%f" % loss, end=' ')
                print("accuracy%f" % accuracy)

            self.backpropagation(data, t)

    def predict(self, data, display=False):
        """
        进行预测，通过softmax给出最后的预测结果和类别所属的概率
        :param data: 预测数据
        :param display: 是否展示softmax的结果
        :return: 预测结果和预测概率
        """
        # 存放每层输出
        Z = []
        x = data
        for j in range(len(self.nodes) - 1):
            x = sigmoid(x.dot(self.W[j]) + self.B[j])
            # 所有层输出存入list中
            Z.append(x)
        softmax_result = softmax(Z[-1])
        result = np.argmax(softmax_result, axis=1)
        rate = []
        for i in range(len(result)):
            rate.append(softmax_result[i, result[i]])
        if display:
            print(softmax_result)
        return result, rate
