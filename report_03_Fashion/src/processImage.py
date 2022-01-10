import cv2.cv2 as cv
import os

import numpy as np

"""
图片做预处理操作

1. 读取特定的图片【后面需要能直接读取所有的图片并且自动做好标签！】
2. 转化为灰度图【done】
3. 对图片的背景做处理【训练一个小网络来做？还是找好用的传统算法？】
3. 填充像素【这里主要是考虑图片被缩放而产生的影响，如果影响不大就不做了】
4. 调整图片分辨率【done】
5. 保存到文件【可能不需要了，写成一个类，先预处理再直接处理，可能更适用一些！】
"""


def read_image(file_pathname, max_size=100):
    """
    读取一个文件夹下面的所有图片
    :param file_pathname: 文件夹路径
    :param max_size: 读取图片的最大数目
    :return: 图片列表
    """
    images = []
    count = 0
    # 遍历该目录下的所有图片文件
    for filename in os.listdir(file_pathname):
        # img = cv.imread(file_pathname + '/' + filename)
        # 如果不专门对颜色做处理的话，直接读入灰度图就行了！
        img = cv.imread(file_pathname + '/' + filename, 0)

        if img is None:
            print("image: %s read failed!" % filename)
            continue

        # 尝试用grabCut提取前景，这个运行起来太慢了！！！
        # img = cv.resize(img, (100, 100))
        #
        # mask = np.zeros(img.shape[:2], dtype=np.uint8)
        # bgdModel = np.zeros((1, 65), dtype=np.float64)
        # fgdModel = np.zeros((1, 65), dtype=np.float64)
        # mask[5:95, 5:95] = 3
        # cv.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv.GC_INIT_WITH_MASK)
        # mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        # ogc = img * mask2[:, :, np.newaxis]
        #
        # cv.imshow("test_real", ogc)

        # 转化为灰度图
        # img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # 进行滤波(这一步影响有多大？)
        # img = cv.GaussianBlur(img, (3, 3), 0)
        # 目前就简单的拉了个阈值来处理背景的问题，有一些图片二值化的效果很差，要尝试一些别的方法！！！
        # _, img = cv.threshold(img, 225, 0, cv.THRESH_TOZERO_INV)
        _, img_binary = cv.threshold(img, 250, 255, cv.THRESH_BINARY_INV)

        # 尝试通过轮廓检测的方式来优化处理结果
        mask = np.ones_like(img_binary)
        # 找到所有的轮廓
        contours, _ = cv.findContours(img_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
        area = []
        # 找到最大的轮廓
        for k in range(len(contours)):
            area.append(cv.contourArea(contours[k]))
        max_idx = np.argmax(np.array(area))

        # 如果轮廓几乎和图片一样大，说明很可能是被阈值化成全白或者全黑的图，应该舍去
        w, h = img_binary.shape[1::-1]
        if area[max_idx] >= 0.9 * w * h:
            continue

        # 填充最大的轮廓
        cv.drawContours(mask, contours, max_idx, 255, cv.FILLED)
        del area

        mask = cv.erode(mask, (3, 3))  # 腐蚀图像
        mask = cv.dilate(mask, (5, 5))  # 膨胀图像

        # TODO 这里应当尝试使用mask来处理原来的灰度图！
        # 这样会处理成白色背景的图片，不太好
        # img_end = cv.add(img, 255 - mask)
        # 直接与运算可能就更好
        img_end = img & mask
        # cv.imshow("img", img)
        # cv.imshow("mask", mask)
        # cv.imshow("img_end", img & mask)
        # cv.waitKey(0)

        # 调整分辨率
        img_end = cv.resize(img_end, (28, 28))

        images.append(img_end)

        count += 1
        if count > max_size:
            break

    return images  # 这里就是28*28的


# 读取成灰度目前还不够，需要调整成1*28*28的，并且还需要归一化！！！
# TODO 这里的归一化和训练集中的归一化是一样的算法吗？这里会不会产生一些问题？
def image_normalize(images):
    """
    对图片进行归一化处理并且增加一个维度（放这里应该不太好？！）
    :param images: 图片列表
    :return: 图片列表
    """
    new_images = []
    for image in images:
        new_image = np.zeros(image.shape, dtype=np.float32)
        cv.normalize(image, new_image, alpha=0, beta=1, norm_type=cv.NORM_MINMAX, dtype=cv.CV_32F)
        new_images.append([new_image])

    return new_images


def get_dataset(data_path, shuffle=False, other_type=False, max_size=100):
    """
    获取数据集，包括图片和标签，可以设置打乱和不打乱
    :param data_path: 数据目录
    :param shuffle: 是否打乱数据集
    :param other_type: 是否读取不支持类别
    :param max_size: 每一个类别的最大数目
    :return: 图片和标签
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
        10: "Others"
    }
    images = []
    labels = []
    if other_type:
        type_num = 11
    else:
        type_num = 10
    for i in range(type_num):
        new_images = read_image(data_path + output_mapping[i], max_size)

        images += new_images
        labels += [i for _ in range(len(new_images))]

        print("Load %d images of: %s" % (len(new_images), output_mapping[i]))

    images = image_normalize(images)

    if shuffle:
        data_size = np.shape(images)[0]
        shuffled_index = np.random.permutation(data_size)
        return np.array(images)[shuffled_index], np.array(labels)[shuffled_index]

    return np.array(images), np.array(labels)


# # 发现如果不这样做，最后使用这个文件中的函数时，会把别的代码也给跑起来！
if __name__ == "__main__":
    _, _ = get_dataset("test/")
    print("test_real finished!")
