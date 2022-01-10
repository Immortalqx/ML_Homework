# 这部分是一段失败的代码，最开始的想法是对矩阵整体做一个二分查找，但是这样的话会少考虑很多情况，导致搜索失败
# 比如3比5小，比1大，按我写的算法，这个只会搜索左上角那个2*2的小矩阵，但3是在matrix[0][2]位置的。
# 后面尝试过二分的时候分别对temp进行+1操作，但是这样就会导致重复查找，并且解决不了上面的问题
# 之后为了弥补少考虑的情况又写了一个searchline函数，但是效果还是不好
#
# def searchline(matrix, target, position, start, end, type):
#     if type == "COL":
#         if start > end:
#             return False
#         temp = (start + end) // 2
#         # print(position, start, end, type, temp, matrix[temp][position])
#         if matrix[temp][position] == target:
#             return True
#         elif matrix[temp][position] < target:
#             return searchline(matrix, target, position, temp + 1, end, type)
#         else:
#             return searchline(matrix, target, position, start, temp - 1, type)
#     elif type == "ROW":
#         if start > end:
#             return False
#         temp = (start + end) // 2
#         # print(position, start, end, type, temp, matrix[position][temp])
#         if matrix[position][temp] == target:
#             return True
#         elif matrix[position][temp] < target:
#             return searchline(matrix, target, position, temp + 1, end, type)
#         else:
#             return searchline(matrix, target, position, start, temp - 1, type)
#     else:
#         return False
#
#
# def searchMatrix(matrix, target, i, j, m, n):
#     if i == m or j == n:
#         return False
#     temp_x = (i + m) // 2
#     temp_y = (j + n) // 2
#
#     # print(i, j, m, n, temp_x, temp_y, matrix[temp_x][temp_y])
#
#     if matrix[temp_x][temp_y] == target:
#         return True
#     elif matrix[temp_x][temp_y] > target:
#         return searchMatrix(matrix, target, i, j, temp_x, temp_y)
#     else:
#         if m - temp_x == 1 and n - temp_y == 1:
#             # print(i, j, m, n, temp_x, temp_y, matrix[temp_x][temp_y], "ififif")
#             return searchline(matrix, target, m, 0, n, "ROW") or \
#                    searchline(matrix, target, n, 0, m, "COL")
#         else:
#             return searchMatrix(matrix, target, temp_x, temp_y, m, n)

# 对一行进行二分查找
def searchline(matrix, target, position, start, end):
    if start > end:
        return False
    temp = (start + end) // 2
    if matrix[temp][position] == target:
        return True
    elif matrix[temp][position] < target:
        return searchline(matrix, target, position, temp + 1, end)
    else:
        return searchline(matrix, target, position, start, temp - 1)


# 通过对每一行进行二分查找来查找矩阵中的元素
# 这个还可以优化，二分查找right的数如果比target大，那么下面的每一行进行查找的时候right都没必要再比它上面那行的right大，可以少查找一些元素
def searchMatrix(matrix, target):
    position = 0
    while position < len(matrix):
        if searchline(matrix, target, position, 0, len(matrix[0]) - 1):
            return True
        else:
            position += 1
    return False


# 简单查找
def searchMatrixSimple(matrix, target):
    index = 0
    while index < len(matrix):
        if target in matrix[index]:
            return True
        index += 1
    return False


if __name__ == '__main__':
    matrix = [
        [1, 4, 7, 11, 15],
        [2, 5, 8, 12, 19],
        [3, 6, 9, 16, 22],
        [10, 13, 14, 17, 24],
        [18, 21, 23, 26, 30]
    ]
    print(searchMatrix(matrix, int(input())))
    # print(searchMatrixSimple(matrix, int(input())))
