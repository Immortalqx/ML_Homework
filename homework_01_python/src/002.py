def fun():
    """直接通过枚举法给出组合结果"""
    i = 0
    for x in range(1, 5):
        for y in range(1, 5):
            for z in range(1, 5):
                if (x != y) and (y != z) and (z != x):
                    i += 1
                    if i % 6:
                        print("%d%d%d" % (x, y, z), end=" | ")
                    else:
                        print("%d%d%d" % (x, y, z))


if __name__ == '__main__':
    fun()
