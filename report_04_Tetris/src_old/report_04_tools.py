# 这里存放各种暂时或者永久的工具代码
import cv2.cv2 as cv
import numpy as np


def get_matrix(state):
    """
    俄罗斯方块的初始state是包括了整个界面的，但我们需要的是游戏界面
    这个函数直接从state中分割出了游戏界面的矩阵，并且进行了阈值化处理
    这个游戏是10*20的，所以最后直接输出10*20的矩阵！
    :param state: game state
    :return: game matrix
    """
    state = cv.cvtColor(state, cv.COLOR_RGB2GRAY)
    _, state = cv.threshold(state, 0, 255, cv.THRESH_BINARY)
    state = state[49:208, 95:176]  # (161, 81)
    # 这里需要推一个计算公式，把163*82的矩阵转化为20*10的矩阵！【done】
    #  已知 方块大小为7*7，163*82
    game_state = np.zeros((20, 10))
    for i in range(20):
        for j in range(10):
            game_state[i, j] = 1 if state[i * 8 + 3, j * 8 + 3] == 255 else 0
    # if show_detail:
    #     print("\n==================================")
    #     print(game_state)
    #     print("==================================\n")
    #     cv.imshow("test", state)
    #     if cv.waitKey(1) == 112:
    #         cv.waitKey(3000)
    return game_state.flatten()


def get_height(matrix):
    """
    计算state中的高度
    :param matrix: state的矩阵
    :return: 最大高度
    """
    matrix = matrix.reshape(20, 10)
    for i in range(20):
        for j in range(10):
            if matrix[i, j] != 0:
                return 20 - i
    return 0  # fixed a bug here!


def get_lines(matrix, min_width=7, type=0):
    """
    计算足够宽的行数
    :param matrix: state的矩阵
    :param min_width: 宽度要求
    :param type: 计算行数的算法
    :return: 行数
    """
    matrix = matrix.reshape(20, 10)
    lines = 0
    for i in range(20):
        w_count = 0
        for j in range(10):
            if matrix[i, j] != 0:
                w_count += 1
        if w_count >= min_width:
            lines += 1 if type == 0 else (w_count - min_width) + 1
    return lines


def get_holes(matrix):
    """
    计算空洞数目（每一列中的空洞数目）
    :param matrix: state的矩阵
    :return: 空洞数目
    """
    matrix = matrix.reshape(20, 10).T
    holes = 0
    for i in range(10):
        count = 0
        for j in range(20):  # 自顶向下遍历
            if matrix[i, j] == 1:  # 找到第一个封口的地方
                while j + 1 < 20 and matrix[i, j + 1] == 0:
                    count += 1
                    j += 1
        holes += count
    return holes


def get_bumpiness(matrix, bumpiness_type=1):
    """
    计算不平整度
    :param matrix: state的矩阵
    :param bumpiness_type: 不平整度的算法类型
    :return: 不平整度
    """
    matrix = matrix.reshape(20, 10).T
    bumpiness = 0
    if bumpiness_type == 0:  # 计算相邻两列之间的差值
        h_last_count = 0
        for i in range(10):
            h_count = 0
            for j in range(20):
                if matrix[i, j] != 0:
                    h_count += 1
            if i == 0:
                h_last_count = h_count
            else:
                bumpiness += abs(h_count - h_last_count)
                h_last_count = h_count
    elif bumpiness_type == 1:  # 计算每列高度与平均高度的差值！
        count = 0
        for i in range(10):
            for j in range(20):
                if matrix[i, j] != 0:
                    count += 1
        count = int(count / 10)
        for i in range(10):
            h_count = 0
            for j in range(20):
                if matrix[i, j] != 0:
                    h_count += 1
            bumpiness += abs(h_count - count)
    return bumpiness


def get_density_difference(matrix):
    """
    计算左右两边密度差，防止左右差距过大！
    :param matrix: state的矩阵
    :return: 密度差
    """
    matrix = matrix.reshape(20, 10)
    l_density = 0
    r_density = 0
    for i in range(20):
        for j in range(10):
            if matrix[i, j] != 0:
                l_density += 1 if j < 5 else 0
                r_density += 1 if j >= 5 else 0
    return abs(l_density - r_density)


def get_well_depth(matrix):
    """
    计算井深（两边是墙或者方块）
    :param matrix: state的矩阵
    :return:井深
    """
    matrix = matrix.reshape(20, 10).T
    well_depth = 0
    for i in range(10):  # 遍历每一列
        for j in range(20):  # 自顶向下
            if matrix[i, j] == 0:
                if (i == 0 or matrix[i - 1, j] == 1) and (i == 9 or matrix[i + 1, j] == 1):
                    well_depth += 1
    return well_depth


# 策略A打算从最简单的奖励开始一步一步往上面添加
def score_policy_A(state, action=None):
    score = 0
    score += get_lines(state, 7, type=1)
    score -= get_bumpiness(state, bumpiness_type=0) * 0.25
    score -= get_bumpiness(state, bumpiness_type=1) * 0.25
    # score -= get_height(state)
    # score -= get_density_difference(state) * 0.2
    return score


# 参考：http://cs231n.stanford.edu/reports/2016/pdfs/121_Report.pdf
# 这个的策略是：
#  1. 惩罚最大高度（最上层方块到底层的距离）
#  2. 奖励宽度大于一定值的行数（在可以消除之后就改为奖励消除行）
#  3. 惩罚空洞（被方块围起来的区域）
#  4. 惩罚不平整度（相邻两列之间的高度差绝对值）
# 相关参数：
# a = -0.510066
# b = 0.760666
# c = -0.35663
# d = -0.184483
def score_policy_B(state, action=None):
    # 设置参数
    a = -0.510066
    b = 0.760666
    c = -0.35663
    d = -0.184483
    # 计算参数参数
    height = get_height(state)  # 最大高度
    lines = get_lines(state, 9)  # 行数
    holes = get_holes(state)  # 空洞
    bumpiness = get_bumpiness(state, 0)  # 不平整度

    score = a * height + b * lines + c * holes + d * bumpiness

    return score


def score_policy_C(state, action=None):
    # 设置参数
    a = -0.510066
    b = 0.760666
    c = -0.35663
    d = -0.184483
    # 计算分数
    score = 0
    score += get_height(state) * a  # 最大高度
    score += get_lines(state, 9) * b  # 行数
    score += (get_holes(state) + get_well_depth(state)) / 2 * c  # 空洞
    score += get_density_difference(state) * d  # 不平整度

    return score


if __name__ == "__main__":
    state = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
                      0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0.,
                      0., 0., 0., 0., 0., 0., 1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0.,
                      0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                      1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0.,
                      0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0.,
                      0., 0., 0., 0., 1., 0., 0., 0.])
    print(score_policy_A(state))
