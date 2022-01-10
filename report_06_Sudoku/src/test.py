import numpy as np
import torch
import torch.nn.functional as F


def test_loss():
    # 假设这是一个3分类问题，一共有4组样本

    # 下面是这个模型的输出
    pred_y = np.array([[0.30722019, -0.8358033, -1.24752918],
                       [0.72186664, 0.58657704, -0.25026393],
                       [0.16449865, -0.44255082, 0.68046693],
                       [-0.52082402, 1.71407838, -1.36618063]])
    pred_y = torch.from_numpy(pred_y)

    # 真实的标签如下所示，很明显这里就是one hot编码
    true_y_one_hot = np.array([[1, 0, 0],
                               [0, 1, 0],
                               [0, 1, 0],
                               [0, 0, 1]])
    true_y_one_hot = torch.from_numpy(true_y_one_hot)

    # 这是采用普通编码的标签
    # true_y_1是正确编码的，对于数独问题，我们认为这里的标签实际上代表1，2，2，3
    true_y_1 = np.array([0, 1, 1, 2])
    true_y_1 = torch.from_numpy(true_y_1)
    # true_y_2是错误编码的，这最后会导致输出类型和标签对应不上
    true_y_2 = np.array([1, 2, 2, 3])
    true_y_2 = torch.from_numpy(true_y_2)
    # target = np.argmax(true_y, axis=1)  # （4,） #其实就是获得每一给类别的整数值，这个和tensorflow里面不一样哦 内部会自动转换为one-hot形式
    # target = torch.LongTensor(target)  # （4,）

    # print(target)  # tensor([0,1,1,2])
    print("-----------------------------------------------------------")

    # 第一步：使用激活函数softmax进行缩放
    print("第一步：使用激活函数softmax进行缩放")
    pred_y = F.softmax(pred_y, dim=1)
    print(pred_y)
    print('-----------------------------------------------------------')

    # 第二步：对每一个缩放之后的值求对数log
    print("第二步：对每一个缩放之后的值求对数log")
    pred_y = torch.log(pred_y)
    print(pred_y)
    print('-----------------------------------------------------------')

    # 第三步：求交叉熵损失
    loss_function = F.nll_loss(pred_y, true_y_1)
    print(loss_function)  # 最终的损失为：tensor(1.5929, dtype=torch.float64)

    print('-----------------------------------------------------------')
    print("两部实现交叉熵")
    # 第一步：直接使用log_softmax,相当于softmax+log
    pred_y = F.log_softmax(pred_y, dim=1)
    print(pred_y)
    print('-----------------------------------------------------------')

    # 第二步：求交叉熵损失
    loss_function = F.nll_loss(pred_y, true_y_1)
    print(loss_function)  # tensor(1.5929, dtype=torch.float64)
    print('-----------------------------------------------------------')

    print("一步实现")
    # 第一步：求交叉熵损失一步到位
    print(pred_y.shape)
    print(true_y_1.shape)
    # 这里如果使用true_y就可以复现之前训练CNN时候的错误
    loss = F.cross_entropy(pred_y, true_y_1)
    print(loss)

    print('-----------------------------------------------------------')


def test_save_path():
    min_loss = 1.2345678
    save_path = "SudokuCNN_model_%.5lf.pth" % min_loss
    print(save_path)


def test_label_acc():
    batch = 1
    a = torch.randn(batch * 81, 9)
    print(torch.max(a, 1)[1])


if __name__ == "__main__":
    test_loss()
