import torch.nn as nn


# TODO
#  这个网络的loss一直在下降，但是准确率一直没有上升
#  我觉得是这个卷积网络没有设置好，毕竟对于拼图游戏来说，边缘比中心更加重要，所以就不应该有太多的池化操作，卷积也要尽量保证边缘的损失足够小
#  目前打算report04 和report 05一起写，哪一个先有进展就做哪个，两个目前都没有比较好的思路。。。实在不行可能会去做report02
class Jigsaw_CNN(nn.Module):
    def __init__(self):
        super(Jigsaw_CNN, self).__init__()
        # 第一层卷积
        # 输入[12,100,100]
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=12, out_channels=32, kernel_size=(5, 5), padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 第二层卷积
        # 输入[32,50,50]
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 5), padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # 第三层卷积
        # 输入[64，25，25]
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(5, 5), padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )

        # 全连接层
        # 输入[32, 25, 25]
        self.fc1 = nn.Linear(in_features=32 * 25 * 25, out_features=2000)
        self.fc2 = nn.Linear(in_features=2000, out_features=600)
        self.fc3 = nn.Linear(in_features=600, out_features=4 * 4)
        self.fc4 = nn.Softmax(dim=2)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        # 向量变化为矩阵（81*9）
        out = out.view(out.shape[0], 4, 4)

        out = self.fc4(out)
        # 这里希望得到的是batch*4*4，最后面的4应该是每一个类型的概率！
        return out
