import numpy as np


def fun(row, col):
    """
    生成一个随机的矩阵，并且用0把该矩阵边界包围起来
    """
    # 不允许0行或者0列的矩阵出现
    if row <= 0 or col <= 0:
        print("ERROR Param!")
        return

    # 生成一个随机矩阵，均匀随机数在[0,1)之间
    # v = np.random.rand(row, col)
    # 生成一个元素全为整数的随机矩阵，方便观察
    v = np.random.randint(0, 100, size=(row, col))

    # 生成一个比v“大一圈”的零矩阵
    vx = np.zeros(tuple(s + 2 for s in v.shape))

    # 这里是把v拷贝到vx中
    vx[tuple(slice(1, -1) for s in v.shape)] = v

    print("row:%d\t" % row + "col:%d" % col)
    print(vx)
    print()


if __name__ == '__main__':
    # 四组测试
    fun(1, 1)
    fun(3, 1)
    fun(1, 3)
    fun(3, 3)
