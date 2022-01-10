from itertools import chain

import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

from torch.autograd import Variable
from torch.utils.data import Dataset

from FashionCNN import FashionCNN
from src.processImage import get_dataset

from fashion_tools import judgeOutLiner
from fashion_tools import display_result

# 尝试使用GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 测试FashionMNIST数据集
def test_demo(display=False, model_path='new_best_model.pth'):
    # 加载数据集
    test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False,
                                                 transform=transforms.Compose([transforms.ToTensor()]))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=100)

    # 加载模型
    model = FashionCNN()
    model.to(device)
    model.load_state_dict(torch.load(model_path))

    total = 0
    correct = 0
    out_liner = 0

    # Lists for knowing classwise accuracy
    predictions_list = []
    labels_list = []

    test_images = None
    test_labels = None
    predictions = None
    for test_images, test_labels in test_loader:
        test_images, test_labels = test_images.to(device), test_labels.to(device)
        labels_list.append(test_labels)

        test = Variable(test_images.view(100, 1, 28, 28))

        outputs = model(test)

        # predictions = torch.max(outputs, 1)[1].to(device)
        predictions, out = judgeOutLiner(outputs.cpu().detach().numpy())
        predictions = torch.from_numpy(predictions).to(device)
        out_liner += out

        predictions_list.append(predictions)

        correct += (predictions == test_labels).sum()
        total += len(test_labels)

    accuracy = correct * 100 / total

    print("Accuracy: {}%".format(accuracy))
    print("不支持类别个数: {}".format(out_liner))

    if display:
        display_result(test_images.view(-1, 28, 28).cpu().detach().numpy(), test_labels.cpu().detach().numpy(),
                       predictions.cpu().detach().numpy(), "test_demo")
        # 画出混淆矩阵
        predictions_l = [predictions_list[i].tolist() for i in range(len(predictions_list))]
        labels_l = [labels_list[i].tolist() for i in range(len(labels_list))]
        predictions_l = list(chain.from_iterable(predictions_l))
        labels_l = list(chain.from_iterable(labels_l))
        cm = confusion_matrix(labels_l, predictions_l)
        plt.matshow(cm)
        plt.colorbar()
        plt.title('Confusion Matrix')
        plt.ylabel('Groundtruth')
        plt.xlabel('Predict')

        # plt.show()


def test_real(display=False, model_path='new_best_model.pth'):
    images, labels = get_dataset("test/", True, True, 3000)
    # 加载模型
    model = FashionCNN()
    model.to(device)
    model.load_state_dict(torch.load(model_path))

    total = 0
    correct = 0
    out_liner = 0

    # Lists for knowing classwise accuracy
    predictions_list = []
    labels_list = []

    test_images = None
    test_labels = None
    predictions = None
    for i in range(len(images) // 100):
        test_images = torch.tensor(images[i * 100:(i + 1) * 100])
        test_labels = torch.tensor(labels[i * 100:(i + 1) * 100])
        labels_list.append(test_labels)
        test_images, test_labels = test_images.to(device), test_labels.to(device)
        test = Variable(test_images.view(100, 1, 28, 28))

        outputs = model(test)
        # predictions = torch.max(outputs, 1)[1].to(device)
        predictions, out = judgeOutLiner(outputs.cpu().detach().numpy())
        predictions = torch.from_numpy(predictions).to(device)
        out_liner += out

        predictions_list.append(predictions)

        correct += (predictions == test_labels).sum()
        total += len(test_labels)

    accuracy = correct * 100 / total
    print("Accuracy: {}%".format(accuracy))
    print("不支持类别个数: {}".format(out_liner))

    if display:
        display_result(test_images.view(-1, 28, 28).cpu().detach().numpy(), test_labels.cpu().detach().numpy(),
                       predictions.cpu().detach().numpy(), "test_real")
        # 画出混淆矩阵
        predictions_l = [predictions_list[i].tolist() for i in range(len(predictions_list))]
        labels_l = [labels_list[i].tolist() for i in range(len(labels_list))]
        predictions_l = list(chain.from_iterable(predictions_l))
        labels_l = list(chain.from_iterable(labels_l))
        cm = confusion_matrix(labels_l, predictions_l)
        plt.matshow(cm)
        plt.colorbar()
        plt.title('Confusion Matrix')
        plt.ylabel('Groundtruth')
        plt.xlabel('Predict')

        # plt.show()


if __name__ == "__main__":
    print("对FashionMNIST测试集进行测试: ")
    test_demo(True, "new_best_model.pth")

    print("\n\n对爬虫得到的数据集进行测试：")
    test_real(True, "new_best_model.pth")

    # 不写到这里的话，就会阻塞对爬虫数据集的测试
    plt.show()
