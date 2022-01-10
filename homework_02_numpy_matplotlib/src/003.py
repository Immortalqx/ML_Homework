import numpy as np


def funA():
    """
    通过两个for循环，利用行坐标和列坐标相加是否为偶数来给矩阵赋1或者0
    """
    # 创建一个8*8的零矩阵
    matrix = np.zeros((8, 8))
    # 如果行坐标+列坐标是偶数就赋值1
    for i in range(8):
        for j in range(8):
            if (i + j) % 2 == 0:
                matrix[i, j] = 1
    return matrix


def funB():
    """
    通过切片来创建一个国际棋盘样式的矩阵，更加简洁
    """
    # 创建一个8*8的零矩阵
    matrix = np.zeros((8, 8))
    # 切片索引，从0行到7行，步长为2，同时从0列到7列，步长为2
    matrix[0:7:2, 0:7:2] = 1
    # 切片索引，从1行到8行，步长为2，同时从1列到8列，步长为2
    matrix[1:8:2, 1:8:2] = 1
    return matrix


if __name__ == '__main__':
    print("方法A：\n"
          "国际棋盘对应的矩阵为：\n", funA())
    print("方法B：\n"
          "国际棋盘对应的矩阵为：\n", funB())
