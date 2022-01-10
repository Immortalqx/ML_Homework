import numpy as np

if __name__ == '__main__':
    # 创建一个随机数组
    v = np.random.randint(0, 10, size=10)
    # 输出原来的数组
    print(v)
    # 方法一
    print(v[::-1])
    # 方法二
    print(np.flipud(v))
