def funA():
    """两个for循环打印乘法表"""
    for i in range(1, 10):
        for j in range(1, i + 1):
            print(f'{j}×{i}={i * j}\t', end='')
        print()


def funB():
    """两个while循环打印乘法表"""
    i = 1
    while i <= 9:
        j = 1
        while j <= i:  # j的大小是由i来控制的
            print(f'{i}×{j}={i * j}', end='\t')
            j += 1
        print('')
        i += 1


def funC(n=1):
    """迭代打印乘法表"""
    if n < 10:
        for m in range(1, n + 1):
            print(f"{m}×{n}={m * n}", end="\t")
        print()
        funC(n + 1)


def funD():
    """通过一行代码打印乘法表"""
    # print('\n'.join([' '.join(["%2s x%2s = %2s" % (j, i, i * j) for j in range(1, i + 1)]) for i in range(1, 10)]))
    print('\n'.join(['\t'.join([f"{j}×{i}={i * j}" for j in range(1, i + 1)]) for i in range(1, 10)]))


if __name__ == '__main__':
    funA()
    funB()
    funC()
    funD()
