import random

# 带有数字、小写字母、大写字母的序列
characters = "0123456789" \
             "qwertyuiopasdfghjklzxcvbnm" \
             "QWERTYUIOPASDFGHJKLZXCVBNM"

if __name__ == '__main__':
    setX = set()
    while len(setX) < 200:
        # 关键是对random的使用，下面是从characters中生成特定长度的序列
        code = "".join(random.sample(characters, 12))
        # 网上的资料说这种方法用于无重复的随机抽样，产生的激活码应该是不会重复的
        # 不过可以通过集合来保证彻底消除重复的激活码
        setX.add(code)
    for m in setX:
        print(m)
