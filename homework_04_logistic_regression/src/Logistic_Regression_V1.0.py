import time
import matplotlib.pyplot as plt
import numpy as np
from math import inf
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# TODO 目前这个是最初的版本，优化版本在V2.0里面
#  要求
#  1. 自己编程实现Logistic Regression的多分类。【done】
#  2. 对比自己实现与sklearn的方法的精度。【done】
#  3. 如何将分类错误的样本可视化出来？【done】
#  想法
#  1.自己实现一个简单的逻辑回归类【done】
#  2.通过OVR解决多分类问题（类别过多的时候OVO会产生非常多的分类器，因此没有考虑）【done】
#  3.对比精度的同时，画出自己的和sklearn的混淆矩阵(Confusion Matrix)【done】
#  4.将图片画出来，用不同颜色标注真值、正确预测和错误预测【done】
#  疑问
#  1.直接把数据打散之后，会不会导致有的类训练样本特别少，这个会造成什么样的影响？？？【我觉得这个会导致结果的随机性更强】
#  2.目前实现的效果很不好，并且速度有点慢，应该如何改进？？？【核心代码需要修改或者重构】
#  3.后面对代码做了一些小修改，准确率从60%~70%提升到85%左右了，但训练要8s，比较久，还是有问题。


def sigmoid(x):
    """
    sigmoid函数
    """
    return 1.0 / (1 + np.exp(-x))


def acc_score(predict_data, raw_data):
    """
    计算结果的准确率
    :param predict_data: 预测值
    :param raw_data: 真值
    """
    cnt = 0
    size = np.shape(predict_data)[0]
    for i in range(size):
        if int(raw_data[i]) == int(predict_data[i]):
            cnt += 1
    score = float(cnt / size)
    return score


fig_num = 0


def display(images, ground_truth, predict, fig_name):
    """
    画出预测结果
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
        else:
            ax.text(6, 7, str(int(predict[i])), color='red', size=20)

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


class Logistic_Regression:
    """
    逻辑回归模型，通过OVR(One-Vs-All)解决多分类问题
    """

    def __init__(self, data, label, max_iter=500, alpha=0.01, tol=0.000005):
        """
        初始化逻辑回归类
        :param data: 加载训练集
        :param label: 加载训练集的标签
        :param max_iter: 最大迭代次数
        :param alpha: 学习率（步长）
        :param tol: 停止求解的标准，当收敛速度过慢时停止求解
        """
        # 保存传入数据
        self.data = data
        self.label = label
        self.max_iter = max_iter
        self.alpha = alpha
        self.tol = tol

        # 进一步处理
        # 数据长度m与单个数据的长度n
        self.m, self.n = np.shape(data)
        # 数据权重
        self.weights = np.ones((10, self.n))
        # 常数项
        self.b = np.ones(10)

    def train(self):
        """
        进行训练，这里采用的是OVR方法解决多分类问题
        """
        # 对于每一个数字
        for number in range(10):
            # 重新打标签
            label = np.copy(self.label)
            for i in range(self.m):
                if label[i] == number:
                    label[i] = 1
                else:
                    label[i] = 0
            # 采用随机梯度上升法进行迭代求解【核心代码】
            for i in range(self.max_iter):
                # 上次的误差
                last_error = inf
                num_index = list(range(self.m))
                for j in range(self.m):
                    # 之前其实是随机打乱了的，所以这里是不是没有必要进行这样的操作？
                    rand_index = int(np.random.uniform(0, len(num_index)))
                    error = label[rand_index] - sigmoid(
                        sum(self.weights[number] * train_data[rand_index]) + self.b[number])
                    if error > last_error:
                        continue
                    self.weights[number] += self.alpha * error * train_data[rand_index]
                    self.b[number] += self.alpha * error
                    del (num_index[rand_index])
                    if abs(last_error - error) < self.tol:
                        break
                    last_error = error

    def predict(self, predict_data):
        """
        进行预测
        :param predict_data: 测试集
        """
        # 保存对测试集的预测结果
        result = np.zeros(np.shape(predict_data)[0])
        # 保存对单个数据，每个分类器给出的结果
        ans = np.zeros(10)
        # 开始对整个测试集进行预测
        for i in range(len(result)):
            # 分别用每个数字的分类器进行预测
            for k in range(10):
                ans[k] = sigmoid(sum(self.weights[k, :] * predict_data[i, :] + self.b[k]))
            ans = list(ans)
            result[i] = ans.index(max(ans))
        return result


if __name__ == "__main__":
    # 加载并且分割各种数据
    train_data, train_target, train_images, test_data, test_target, test_images = load_data(train_size=0.75)

    # =====================使用自己的方法做预测========================
    # 生成一个逻辑回归类实例，并且把训练集导入
    lr = Logistic_Regression(train_data, train_target)
    # 开始训练并且计时
    start = time.clock()
    lr.train()
    end = time.clock()
    print("训练用时:\t%f s" % (end - start))
    # 对训练集做预测并且输出准确度与结果
    train_result = lr.predict(train_data)
    print("对训练集做预测:\t", acc_score(train_result, train_target))
    display(train_images, train_target, train_result, "My Train")
    # 对测试集做预测并且输出准确度与结果
    test_result = lr.predict(test_data)
    print("对测试集做预测:\t", acc_score(test_result, test_target))
    display(test_images, test_target, test_result, "My Test")

    # =====================使用sklearn做预测========================
    skl = LogisticRegression()

    start = time.clock()
    skl.fit(train_data, train_target)
    end = time.clock()
    print("训练用时:\t%f s" % (end - start))

    pred_train = skl.predict(train_data)
    print("对训练集做预测:\t", acc_score(pred_train, train_target))
    display(train_images, train_target, pred_train, "Sklearn Train")

    pred_test = skl.predict(test_data)
    print("对测试集做预测:\t", acc_score(pred_test, test_target))
    display(test_images, test_target, pred_test, "Sklearn Test")

    plt.show()
