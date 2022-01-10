import torch.nn as nn


class SudokuCNN(nn.Module):
    def __init__(self):
        super(SudokuCNN, self).__init__()
        # 第一层卷积
        # 输入[1,9,9]
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # 9-3+2*1+1
        # 第二层卷积
        # 输入[64,9,9]
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # 第三层卷积
        # 输入[64，9，9]
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        # 全连接层
        # 输入[128, 9, 9]
        self.fc1 = nn.Linear(in_features=128 * 9 * 9, out_features=1000)
        self.fc2 = nn.Linear(in_features=1000, out_features=81 * 9)
        self.fc3 = nn.Softmax(dim=2)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        # 矩阵展开为向量
        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.fc2(out)

        # 向量变化为矩阵（81*9）
        out = out.view(out.shape[0], 81, 9)

        out = self.fc3(out)
        return out
