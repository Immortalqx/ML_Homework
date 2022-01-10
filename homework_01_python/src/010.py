import os


# 在指定目录下统计所有的py文件、c文件、cpp文件，以列表形式返回
def collect_files(dir):
    py_fileList = []
    c_fileList = []
    for parent, dirnames, filenames in os.walk(dir):
        for filename in filenames:
            if filename.endswith(".py"):
                py_fileList.append(os.path.join(parent, filename))
            elif filename.endswith(".c") or filename.endswith(".cpp"):
                c_fileList.append(os.path.join(parent, filename))
    return py_fileList, c_fileList


# 计算单个文件内的代码行数，包括了注释和空行
def calc_linenum(file):
    with open(file) as fp:
        content_list = fp.readlines()
        linenum = len(content_list)
    return linenum


if __name__ == '__main__':
    # 定义代码所在的目录
    base_path = "/home/lqx/Test"

    py_files, c_files = collect_files(base_path)

    py_linenum = 0
    c_linenum = 0

    for f in py_files:
        linenum = calc_linenum(f)
        py_linenum += linenum
    for f in c_files:
        linenum = calc_linenum(f)
        c_linenum += linenum

    print("Python代码总行数为:\t", py_linenum)
    print("C代码总行数为:\t", c_linenum)
