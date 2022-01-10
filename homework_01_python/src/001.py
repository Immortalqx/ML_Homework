import string


def getText():
    """获取文章信息并且做初步的处理操作"""
    # 从终端读取文章
    txt = input()
    # 全部转化为小写
    txt.lower()
    # 处理标点符号
    temp = []
    for c in txt:
        if c not in string.punctuation:
            temp.append(c)
        else:
            temp.append(' ')
    newTxt = "".join(temp)
    # print(newTxt)
    return newTxt


def getWords(txt):
    """从初步处理的文章中分离出单词"""
    # 按照空格分割
    words = txt.split()
    return words


def countWords(words):
    """对单词进行计数操作"""
    counts = {}
    for word in words:
        if word in counts:
            counts[word] += 1
        else:
            counts[word] = 1
    return counts


def printResult(counts):
    """字典转换成列表排序，输出最后的结果"""
    items = list(counts.items())
    items.sort(key=lambda x: x[1], reverse=True)
    for i in range(len(counts)):
        word, count = items[i]
        # print("{0:<10}{1:>5}".format("|" + word + "|", count))
        print("{0:<10}{1:>5}".format(word, count))


if __name__ == '__main__':
    txt = getText()
    words = getWords(txt)
    counts = countWords(words)
    printResult(counts)
