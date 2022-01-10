import numpy as np
import matplotlib.pyplot as plt


def function(x):
    """
    定义一个一元二次函数
    """
    return x ** 2 + 2 * x + 5


def draw(ladder_num):
    """
    画出一个二次函数，同时画出梯形法求积分时的各个梯形
    :param ladder_num: 梯形的个数
    """
    # 在-10,10之间返回100个均匀间隔的数字
    x = np.linspace(-10, 10, num=100)
    y = function(x)
    # 画出二次函数图形
    plt.plot(x, y, color="red", linewidth=3)

    a = np.linspace(-10, 10, num=ladder_num)
    # 画梯形的上底和下底
    for i in range(ladder_num):
        # 不指定颜色的话会画出彩色的线条来
        # plt.plot([a[i], a[i]], [0, function(a[i])])
        plt.plot([a[i], a[i]], [0, function(a[i])], color="gray")

    ladders = [];
    # 因为梯形的腰是呈一条直线，所以这里存下各点坐标
    for i in range(ladder_num):
        ladders.append([a[i], function(a[i])])

    npladders = np.array(ladders)
    # 把梯形的斜腰连起来
    plt.plot(npladders[:, 0], npladders[:, 1], color="gray")
    plt.show()


if __name__ == '__main__':
    draw(30)
