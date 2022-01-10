import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn import svm

from torch.utils.data import Dataset

# TODO 尝试把这个OCSVM弄起来！！！

# 尝试使用GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载数据集
test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False,
                                             transform=transforms.Compose([transforms.ToTensor()]))
test_loader = torch.utils.data.DataLoader(test_set, batch_size=100)


# 测试FashionMNIST数据集
def test_demo():
    try:
        # 加载数据集
        test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False,
                                                     transform=transforms.Compose([transforms.ToTensor()]))
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=100)

        # 模型拟合
        clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)

        for test_images, test_labels in test_loader:
            print(np.shape(test_images))
            clf.fit(test_images)
            print(clf.predict(test_images))
            # n_error_train = y_pred_train[y_pred_train == -1].size
    finally:
        print("OCSVM failed")

# def judgeOutLiner(outputs):
#     labels = []
#     for i in range(len(outputs)):
#         label = outputs[i]
#         if np.max(label) - np.min(label) < 0.3:
#             # print("out liners!")
#             labels.append(-1)
#         else:
#             # print("type is:", np.argmax(label))
#             labels.append(np.argmax(label))
#     # print(labels)
#     return np.array(labels)

# if __name__ == "__main__":
#     test_demo()
