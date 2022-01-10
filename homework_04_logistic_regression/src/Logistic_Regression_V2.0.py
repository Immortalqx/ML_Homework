import random
import time
import matplotlib.pyplot as plt
import numpy as np
from math import inf
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix


# Question
#  1.这里使用了随机梯度上升法(SGD)，就稍微修改了课上讲的那个方法，但是这里实际上这里每一次迭代使用了所有的样本来进行梯度的更新，
# 里面随机取值也没有意义，所以这里实际上是批量梯度下降法？
# Notes
#  1.折腾了两三天，不过差不多把这一块的内容都搞清楚了
#  2.最开始的版本效果不好、时间比较长是因为没有考虑error是可以正可以负的，本来是希望梯度下降一次如果error变大了就跳过更新，但实际上起了负面的效果；
# 不过其他的问题还是没照出来，感觉只能做到85%左右的正确率，然后结果的随机性还是比较大的，有时候会突然降到60%多。
#  3.感觉还是没把准确率提升上去


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

    def train(self, method="SGD"):
        """
        进行训练，这里采用的是OVR方法解决多分类问题
        :param method: 使用的求解方法
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
            if method == "SGD":
                # 采用随机梯度上升法进行迭代求解
                self.SGD(number, label)
            elif method == "MBGD":
                # 采用小批量梯度下降法
                self.MBGD(number, label)
            else:
                print("error method:", method)
                exit(-1)

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

    def SGD(self, number, label):
        """
        使用老师上课讲的随机梯度上升法求解
        :param number: 目前需要被分类的数字
        :param label: 根据这个数组制作的标签
        """
        for i in range(self.max_iter):
            # 上次的误差
            last_error = inf
            num_index = list(range(self.m))
            for j in range(self.m):
                # FIXME
                #  之前其实是随机打乱了的，所以这里是不是没有必要进行这样的操作？
                #  这里实际上还是用上了所有的样本，所以实际上这里随机取值对结果并没有影响
                rand_index = int(np.random.uniform(0, len(num_index)))
                error = label[rand_index] - sigmoid(
                    sum(self.weights[number] * train_data[rand_index]) + self.b[number])
                # 实际上下面这样做的效果不好，error可能为正也可能为负。。。
                # if abs(error) > abs(last_error):
                # if error > last_error:
                #     continue
                # 下面右边应该是 学习率*(1-g(z))*z，是求偏导的过程
                self.weights[number] += self.alpha * error * train_data[rand_index]
                self.b[number] += self.alpha * error
                del (num_index[rand_index])
                if abs(last_error - error) < self.tol:
                    break
                last_error = error

    def MBGD(self, number, label):
        """
        尝试使用小批量梯度下降法求解
        :param number: 目前需要被分类的数字
        :param label: 根据这个数组制作的标签
        """
        for i in range(self.max_iter):
            # 上次的误差
            last_error = inf
            # 小批量的数目
            if self.m < 200:
                m = random.randint(1, self.m)
            else:
                m = random.randint(10, 200)
            num_index = list(range(self.m))
            for j in range(m):
                rand_index = int(np.random.uniform(0, len(num_index)))
                error = label[rand_index] - sigmoid(
                    sum(self.weights[number] * train_data[rand_index]) + self.b[number])
                self.weights[number] += self.alpha * error * train_data[rand_index]
                self.b[number] += self.alpha * error
                del (num_index[rand_index])
                if abs(last_error - error) < self.tol:
                    break
                last_error = error


if __name__ == "__main__":
    # 加载并且分割各种数据
    train_data, train_target, train_images, test_data, test_target, test_images = load_data(train_size=0.75)

    # =====================使用自己的方法做预测========================
    # 生成一个逻辑回归类实例，并且把训练集导入
    lr = Logistic_Regression(train_data, train_target, 3000, 0.01, 0.0000005)
    # 开始训练并且计时
    start = time.clock()
    lr.train(method="SGD")
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
