import torch.nn as nn

"""
Notes:

    torch.nn是专门为神经网络设计的模块化接口。nn构建于autograd之上，可以用来定义和运行神经网络。
    nn.Module是nn中十分重要的类,包含网络各层的定义及forward方法。
    定义自已的网络：
        需要继承nn.Module类，并实现forward方法。
        一般把网络中具有可学习参数的层放在构造函数__init__()中，
        不具有可学习参数的层(如ReLU)可放在构造函数中，也可不放在构造函数中(而在forward中使用nn.functional来代替)
        
        只要在nn.Module的子类中定义了forward函数，backward函数就会被自动实现(利用Autograd)。
        在forward函数中可以使用任何Variable支持的函数，毕竟在整个pytorch构建的图中，是Variable在流动。还可以使用
        if,for,print,log等python语法.
        
        注：Pytorch基于nn.Module构建的模型中，只支持mini-batch的Variable输入方式，
        比如，只有一张输入图片，也需要变成 N x C x H x W 的形式：(batch channel H W) 
        
        input_image = torch.FloatTensor(1, 28, 28)
        input_image = Variable(input_image)
    input_image = input_image.unsqueeze(0)   # 1 x 1 x 28 x 28
"""


class FashionCNN(nn.Module):

    # 在设计的时候，注意CNN里面的输入输出公式！！！
    def __init__(self):
        super(FashionCNN, self).__init__()

        # 第一层卷积
        # 输入[1,28,28]
        self.layer1 = nn.Sequential(
            # 卷积层
            #   padding是边缘填充，可以分为四类：零填充，常数填充，镜像填充，重复填充。
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            # 归一化处理https://zhuanlan.zhihu.com/p/205453986
            #   在卷积神经网络的卷积层之后总会添加BatchNorm2d进行数据的归一化处理
            #   这使得数据在进行Relu之前不会因为数据过大而导致网络性能的不稳定
            nn.BatchNorm2d(32),
            nn.ReLU(),
            # 池化层
            # 　池化层是夹在连续的卷积层的中间层，池化层可以非常有效地缩小矩阵的尺寸。从而减少最后全连接层中的参数。
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 第二层卷积
        # 输入[32,14,14]https://img-blog.csdnimg.cn/20200604155746947.bmp
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=128, kernel_size=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # 第三层卷积
        # 输入[128，6，6]
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 防止过拟合
        self.drop = nn.Dropout2d(0.25)

        # 全连接层
        # 输入[64,2，2]
        self.fc1 = nn.Linear(in_features=64 * 2 * 2, out_features=600)
        self.fc2 = nn.Linear(in_features=600, out_features=100)
        self.fc3 = nn.Linear(in_features=100, out_features=10)
        self.fc4 = nn.Softmax()

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.fc4(out)

        return out
