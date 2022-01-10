def funA(I):
    if I <= 10:
        return I * 0.1
    elif I < 20:
        return 10 * 0.1 + (I - 10) * 0.075
    elif I < 40:
        return 10 * 0.1 + 10 * 0.075 + (I - 20) * 0.05
    elif I < 60:
        return 10 * 0.1 + 10 * 0.075 + 20 * 0.05 + (I - 40) * 0.03
    elif I < 100:
        return 10 * 0.1 + 10 * 0.075 + 20 * 0.05 + 20 * 0.03 + (I - 60) * 0.015
    else:
        return 10 * 0.1 + 10 * 0.075 + 20 * 0.05 + 20 * 0.03 + 40 * 0.015 + (I - 100) * 0.01


# def funB(I):
#     TODO 暂时没思路

if __name__ == '__main__':
    print("请输入当月利润，单位：万元")
    print("I=", end='')
    print("\n" + "应发放奖金总数为：", funA(int(input())))
