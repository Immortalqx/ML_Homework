# 这个文件用来存放各种各样的工具函数，不然东一个西一个也太乱了
import numpy as np
import cv2.cv2 as cv
from matplotlib import pyplot as plt


def get_label(index):
    """
    通过类型索引获取类名
    :param index: 索引
    :return:类名
    """
    output_mapping = {
        0: "T-shirt",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle Boot",
        10: "Not Support"
    }
    return output_mapping[index]


fig_num = 0


def display_result(images, ground_truth, predict, fig_name):
    """
    画出预测结果
    :param images: 数字的图片
    :param ground_truth: 真值
    :param predict: 预测值
    :param fig_name: 图片名字
    """
    # plot the digits
    global fig_num
    fig_num += 1
    if fig_name is None:
        fig = plt.figure(fig_num, figsize=(6, 6))  # figure size in inches
    else:
        fig = plt.figure(fig_name, figsize=(6, 6))  # figure size in inches
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    size = np.shape(predict)[0]
    if size > 56:
        size = 56
    # plot the digits: each image is 8x8 pixels
    for i in range(size):
        ax = fig.add_subplot(8, 8, i + 1, xticks=[], yticks=[])
        ax.imshow(cv.cvtColor(images[i], cv.COLOR_BGR2RGB))
        # label the image with the target value
        if int(ground_truth[i]) == int(predict[i]):
            ax.text(0, 7, get_label(predict[i]), color='green', size=12)
        else:
            ax.text(0, 7, get_label(predict[i]), color='red', size=12)


def judgeOutLiner(outputs, sigma=0.5, threshold1=0.9, threshold2=0.5):
    """
    通过双阈值进行离群点检测
    :param outputs: 预测结果
    :param sigma: 距离均值的距离
    :param threshold1: 阈值1
    :param threshold2: 阈值2
    :return:
    """
    labels = []
    sum = 0
    for i in range(len(outputs)):
        label = outputs[i]
        if (np.max(label) - np.mean(label)) < sigma and np.max(label) < threshold1 or np.max(label) < threshold2:
            # print("out liners!")
            labels.append(10)
            sum += 1
        else:
            # print("type is:", np.argmax(label))
            labels.append(np.argmax(label))
    # print(labels)
    return np.array(labels), sum
