import random
import matplotlib.pyplot as plt
import numpy as np
from math import sin, cos


def funA():
    """
    这是只有四种走法的版本，醉汉只会从上下左右中选择一种
    """
    plt.figure(1)
    # 设置[0,0]为起点
    start = [0, 0]
    # 定义上、下、左、右四种走法     【右上、右下、左上、左下八种走法】X
    # step = sqrt(2) / 2
    # p = np.array([[0, 1], [0, -1], [1, 0], [-1, 0],[step, step], [step, -step], [-step, step], [-step, -step]])
    p = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    # 定义路径
    path = [start, ]
    # 设置醉汉会走1000步
    while len(path) < 1000:
        # 在四种走法中随机选择一种走法
        b = np.random.randint(len(p))
        # 更新起点
        start = np.array(start) + p[b]
        # 向路径中插入新的节点
        c = list(start)
        path.append(c)

    # 转置并且用红色画出起点、绿色画出路径
    result = np.array(path).T
    plt.scatter(0, 0, c='r')
    plt.plot(result[0], result[1], c='g')
    # plt.show()


def funB():
    """
    在这个版本中，醉汉会向任意方向运动，步长固定为1
    """
    plt.figure(2)
    # 设置[0,0]为起点
    start = [0, 0]
    # 定义路径
    path = [start, ]
    # 设置醉汉会走1000步
    while len(path) < 1000:
        # 从0到2Pi随机取一个值作为醉汉这一步的方向
        angle = random.uniform(0, 2 * 3.14159)
        # 计算醉汉本次的路径
        step = [1 * sin(angle), 1 * cos(angle)]
        # 更新起点
        start = np.array(start) + step
        # 向路径中插入新的节点
        c = list(start)
        path.append(c)

    # 转置并且用红色画出起点、绿色画出路径
    result = np.array(path).T
    plt.scatter(0, 0, c='r')
    plt.plot(result[0], result[1], c='g')
    # plt.show()


def funC():
    """
    funA函数的改进版，醉汉在二维空间中游走，用z表示步子
    """
    plt.figure(3)
    ax3d = plt.axes(projection='3d')
    # 设置[0,0]为起点
    start = [0, 0]
    # 定义上、下、左、右四种走法
    p = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    # 定义路径
    path = [start, ]
    # 设置醉汉会走1000步
    while len(path) < 1000:
        # 在四种走法中随机选择一种走法
        b = np.random.randint(len(p))
        # 更新起点
        start = np.array(start) + p[b]
        # 向路径中插入新的节点
        c = list(start)
        path.append(c)

    # 转置并且用红色画出起点、绿色画出路径
    result = np.array(path).T
    step = np.arange(len(result[0]))
    ax3d.scatter3D(0, 0, 0, c='r')
    ax3d.plot3D(result[0], result[1], step, c='g')
    # plt.show()


if __name__ == '__main__':
    funA()
    funB()
    funC()
    plt.show()
