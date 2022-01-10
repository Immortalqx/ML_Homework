import glob
import os


# 方法一
# 主要思路：如果是文件，则记录文件名称，如果是文件夹则进行递归操作，直到获取该文件夹下所有文件
def getAllFileA(path, fileList):
    # 存放文件夹路径的列表
    dirList = []
    for ff in os.listdir(path):
        wholePath = os.path.join(path, ff)
        # 如果是文件添加到结果文件列表中
        if os.path.isdir(wholePath):
            dirList.append(wholePath)
        # 如果是文件夹，存到文件夹列表中
        if os.path.isfile(wholePath):
            fileList.append(wholePath)  #
    # 递归，让所有的文件路径都会被保存在这个列表中
    for dir in dirList:
        getAllFileA(dir, fileList)


# 从文件路径列表中筛选出指定后缀的文件
def getKeyFileA(fileList, keyword):
    for ff in fileList[:]:
        if not ff.endswith(keyword):
            fileList.remove(ff)


def funA(path, keyword):
    fileList = []
    getAllFileA(path, fileList)
    getKeyFileA(fileList, keyword)
    print("对应文件总数为:\t", len(fileList))
    for res in fileList:
        print(res)


# 方法二
# 主要思路：使用os.walk。
# os.walk() 方法用于通过在目录树中游走输出在目录中的文件名，向上或者向下，返回的是一个三元组(root,dirs,files)
#   root 所指的是当前正在遍历的这个文件夹的本身的地址
#   dirs 是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
#   files 同样是 list , 内容是该文件夹中所有的文件(不包括子目录)
def funB(path, keyword):
    fileList = []
    for home, dirs, files in os.walk(path):
        # 获得所有文件名称
        for filename in files:
            if filename.endswith(keyword):
                fileList.append(os.path.join(home, filename))

    print("对应文件总数为:\t", len(fileList))
    for res in fileList:
        print(res)


# 方法三：借助glob方法
def funC(path, keyword):
    keyList = []
    for home, dirs, files in os.walk(path):
        file = home + "/*" + keyword
        for filename in glob.glob(file):
            keyList.append(filename)
    print("对应文件总数为:\t", len(keyList))
    for res in keyList:
        print(res)


if __name__ == '__main__':
    # 查找地址
    path = r'/home/lqx/Lab/Test'
    # 关键字
    keyword = '.txt'

    # 方法一：
    funA(path, keyword)
    # 方法二：
    funB(path, keyword)
    # 方法三
    funC(path, keyword)
