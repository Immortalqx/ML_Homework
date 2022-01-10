import torch.utils.data as data
import torch
import pandas as pd
import cv2.cv2 as cv


def create_tensors(input_data, train_split=0.7):
    """
    分割训练集和测试集
    :param input_data: 待分割的数据
    :param train_split: 分割的比例
    :return: 训练集，测试集
    """
    # 数据的长度
    data_size = input_data.shape[0]

    # def encode_image(s):
    #     s = s.strip('.jpg')
    #     zeros = torch.zeros((1, 1), dtype=torch.int)
    #     zeros[0, 0] = int(s)
    #     return zeros

    def encode_image(s):
        img = cv.imread("../data/puzzle_2x2/train/" + s)
        zeros = torch.zeros((3, 200, 200), dtype=torch.float)
        for channel in range(3):
            zeros[channel, :, :] = torch.tensor(img[:, :, channel])
        return zeros

    def encode_label(s):
        zeros = torch.zeros((1, 4), dtype=torch.int)
        for a in range(4):
            zeros[0, a] = int(s[a * 2])
        return zeros

    # 得到编码内容
    image_t = input_data.image.apply(encode_image)
    label_t = input_data.label.apply(encode_label)

    # 将编码好的内容拼接起来
    image_t = torch.cat(image_t.values.tolist())
    image_t = image_t.view(-1, 3, 200, 200)
    label_t = torch.cat(label_t.values.tolist())

    # 按比例进行随机分割
    randperm = torch.randperm(data_size)
    train = randperm[:int(train_split * data_size)]
    test = randperm[int(train_split * data_size):]

    # 打包训练集和标签
    return data.TensorDataset(image_t[train], label_t[train]), \
           data.TensorDataset(image_t[test], label_t[test])


def load_dataset(filepath, subsample=10000):
    """
    加载数据集
    :param filepath: 数据集文件
    :param subsample: 数据集总共的行数
    :return: 训练集、测试集
    """

    dataset = pd.read_csv(filepath, sep=',')
    # 返回随机 subsample 行数据
    my_sample = dataset.sample(subsample)
    # 分割出训练集和测试集
    train_set, test_set = create_tensors(my_sample)
    return train_set, test_set


def split_image(image):
    """
    分割图片，输入batch*3*200*200的，根据通道和位置分割成12*100*100的
    :param image: 待分割图片
    :return: 分割好的图片
    """
    image_A = image[:, :, :100, :100]  # 左上
    image_B = image[:, :, :100, 100:]  # 右上
    image_C = image[:, :, 100:, :100]  # 左下
    image_D = image[:, :, 100:, 100:]  # 右下
    image_split = torch.cat((image_A, image_B, image_C, image_D))

    return image_split.view(image.shape[0], 12, 100, 100)


def TEST_load_dataset():
    train_set, test_set = load_dataset("../data/puzzle_2x2/train.csv", 10)
    for train_quiz, train_label in train_set:
        train_quiz = train_quiz.view(1, 3, 200, 200)
        print(train_quiz.shape)
        # print(train_image)
        print(train_label.shape)
        # print(train_label)
        print(split_image(train_quiz).shape)
        break


def TEST_split_image():
    image = cv.imread("../data/puzzle_2x2/train/0.jpg")
    image_A = image[:100, :100, :]  # 左上
    image_B = image[:100, 100:, :]  # 右上
    image_C = image[100:, :100, :]  # 左下
    image_D = image[100:, 100:, :]  # 右下
    cv.imshow("test", image)
    cv.imshow("image_A", image_A)
    cv.imshow("image_B", image_B)
    cv.imshow("image_C", image_C)
    cv.imshow("image_D", image_D)
    cv.waitKey(0)


if __name__ == "__main__":
    # TEST
    TEST_load_dataset()
    # TEST_split_image()
