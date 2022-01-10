import numpy as np

if __name__ == '__main__':
    M = np.random.randint(-100, 100, size=(10, 10))
    print("10*10随机数组为:\n", M)
    print("最大值为:\t%d\t" % M.max())
    print("最小值为:\t%d\t" % M.min())
