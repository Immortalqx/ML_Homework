# 方法一：
def funA():
    i = 2
    sum = 0
    while i < 101:
        if i % 2 == 0:
            sum += i
        else:
            sum -= i
        i = i + 1
    return sum


if __name__ == '__main__':
    # 方法一
    print(funA())

    # 方法二
    # 使用sum方法，一行写完
    print(sum([i * (-1) ** (i % 2) for i in range(2, 101)]))
