# 这里存放各种暂时或者永久的工具代码
import numpy as np


# 在下面的代码中，如果设置为!=0就意味着考虑为下落的方块，如果是==0就是仅考虑已经放置的方块
# 下落的方块为2
# 已经放置的方块为1
# 空为0


def get_height(matrix):
    """
    计算state中的高度，仅仅考虑已经放置的方块！
    :param matrix: state的矩阵
    :return: 最大高度
    """
    matrix = matrix.reshape(21, 10)
    for i in range(21):
        for j in range(10):
            if matrix[i, j] == 0:
                return 21 - i
    return 0  # fixed a bug here!


def get_lines(matrix, min_width=7, type=0):
    """
    计算足够宽的行数，仅仅考虑已经放置的方块！
    :param matrix: state的矩阵
    :param min_width: 宽度要求
    :param type: 计算行数的算法
    :return: 行数
    """
    matrix = matrix.reshape(21, 10)
    lines = 0
    for i in range(21):
        w_count = 0
        for j in range(10):
            if matrix[i, j] == 0:
                w_count += 1
        if w_count >= min_width:
            lines += 1 if type == 0 else (w_count - min_width) + 1
    return lines


def get_holes(matrix):
    """
    计算空洞数目（每一列中的空洞数目）
    仅考虑已经放置的方块
    :param matrix: state的矩阵
    :return: 空洞数目
    """
    matrix = matrix.reshape(21, 10).T
    holes = 0
    for i in range(10):
        count = 0
        for j in range(21):  # 自顶向下遍历
            if matrix[i, j] == 0:  # 找到第一个封口的地方
                while j + 1 < 21 and matrix[i, j + 1] == 0:
                    count += 1
                    j += 1
        holes += count
    return holes


def get_bumpiness(matrix, bumpiness_type=1):
    """
    计算不平整度，仅仅考虑已经放置的方块
    :param matrix: state的矩阵
    :param bumpiness_type: 不平整度的算法类型
    :return: 不平整度
    """
    matrix = matrix.reshape(21, 10).T
    bumpiness = 0
    if bumpiness_type == 0:  # 计算相邻两列之间的差值
        h_last_count = 0
        for i in range(10):
            h_count = 0
            for j in range(21):
                if matrix[i, j] == 0:
                    h_count += 1
            if i == 0:
                h_last_count = h_count
            else:
                bumpiness += abs(h_count - h_last_count)
                h_last_count = h_count
    elif bumpiness_type != 0:  # 计算每列高度与平均高度的差值！
        count = 0
        for i in range(10):
            for j in range(21):
                if matrix[i, j] == 0:
                    count += 1
        count = int(count / 10)
        for i in range(10):
            h_count = 0
            for j in range(21):
                if matrix[i, j] != 0:
                    h_count += 1
            bumpiness += abs(h_count - count)
    return bumpiness


def get_density_difference(matrix):
    """
    计算左右两边密度差，防止左右差距过大！
    仅考虑已经放置的方块！
    :param matrix: state的矩阵
    :return: 密度差
    """
    matrix = matrix.reshape(21, 10)
    l_density = 0
    r_density = 0
    for i in range(21):
        for j in range(10):
            if matrix[i, j] == 0:
                l_density += 1 if j < 5 else 0
                r_density += 1 if j >= 5 else 0
    return abs(l_density - r_density)


def get_well_depth(matrix):
    """
    计算井深（两边是墙或者方块）
    仅考虑已经放置的方块
    :param matrix: state的矩阵
    :return:井深
    """
    matrix = matrix.reshape(21, 10).T
    well_depth = 0
    for i in range(10):  # 遍历每一列
        for j in range(21):  # 自顶向下
            if matrix[i, j] == 0:
                if (i == 0 or matrix[i - 1, j] != 0) and (i == 9 or matrix[i + 1, j] != 0):
                    well_depth += 1
    return well_depth


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
def score_policy_A(state, action=None):
    # 设置参数
    a = -0.510066
    b = 0.760666
    c = -0.35663
    d = -0.184483
    # 计算参数参数
    height = get_height(state)  # 最大高度
    lines = get_lines(state, 9, type=0)  # 行数
    holes = (get_holes(state))  # 空洞
    bumpiness = (get_bumpiness(state, 0))  # 不平整度

    score = a * height + b * lines + c * holes + d * bumpiness

    return score


def score_policy_B(state, action=None):
    # 设置参数
    a = -0.510066
    b = 0.760666
    c = -0.35663
    d = -0.184483
    # 计算参数参数
    height = get_height(state)  # 最大高度
    lines = get_lines(state, 7, type=1)  # 行数
    holes = (get_holes(state) + get_well_depth(state)) * 0.5  # 空洞
    bumpiness = (get_bumpiness(state, 0) + get_bumpiness(state, 1)) * 0.5  # 不平整度

    score = a * height + b * lines + c * holes + d * bumpiness
    return score


def score_policy_C(state, action=None):
    score = 0
    score -= get_height(state)
    score += get_lines(state)
    score -= get_density_difference(state) * 0.3
    return score


# TODO 设计一个不太一样的奖励函数，最好不要根据前面的来！(最后再尝试一下，没其他时间了)
#  目前是在Tetris里面设计了一个reward，如果效果不好再过来设计！
def score_policy_END():
    pass


if __name__ == "__main__":
    state = np.array(
        [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
         0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
         1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
         1., 1., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 0., 0., 0., 0., 0.,
         0., 0., 0., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0.,
         0., 0., 0., 0., 0., 1., 1., 1., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1.,
         1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0.])
    print(score_policy_B(state))
