from math import sin, exp
import numpy as np
import matplotlib.pyplot as plt


def function(x):
    """
    定义一个一元二次函数
    """
    return (sin(x - 2)) ** 2 * exp(-x ** 2)


def draw():
    x = np.linspace(0, 2, num=100)
    # function不支持给如列表并计算出列表来
    y = []
    for i in x:
        j = function(i)
        y.append(j)
    # 画出二次函数图形
    plt.plot(x, y, linewidth=2)
    plt.xlabel("x label")
    plt.ylabel("y label")
    plt.title("$f(x) = sin^2(x - 2) e^{-x^2}$")

    plt.show()


if __name__ == '__main__':
    draw()
