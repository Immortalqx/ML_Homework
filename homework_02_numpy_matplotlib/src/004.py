import numpy as np


# 最简单的方法
def funA(A, b):
    """
    用 numpy.linalg.solve 求解
    """
    # 通过numpy.linalg.solve求解
    try:
        r = np.linalg.solve(A, b)
        print("方法A：\n", r)
    except:
        print("方程无解！")


# 方法B的精度应该没有方法A高
def funB(A, b):
    """
    通过矩阵求逆来解多元线性方程
    """
    try:
        At = np.linalg.inv(A)
        r = At.dot(b)
        print("方法B：\n", r)
    except:
        print("方程无解！")


# TODO 消元法求解方程组比较复杂
def funC(A, b):
    """
    通过消元法解方程
    """
    pass


if __name__ == '__main__':
    # 构造系数矩阵A
    A = np.array([[3, 4, 2], [5, 3, 4], [8, 2, 7]])
    # A = np.mat("3,4,2,5,6;"
    #            "3,111,345,4,2;"
    #            "8,123,11,45,7;"
    #            "1,2,3,4,5;"
    #            "2,3,4,5,6")
    # 构造转置矩阵b
    b = np.array([10, 14, 20]).T
    # b = np.array([10, 14, 20, 221, 1243]).T

    funA(A, b)
    funB(A, b)
    funC(A, b)
