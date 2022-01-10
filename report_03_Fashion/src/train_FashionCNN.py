import copy
from itertools import chain

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable
from torch.utils.data import Dataset

from FashionCNN import FashionCNN

# from processImage import image_normalize

# 尝试使用GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载数据集
train_set = torchvision.datasets.FashionMNIST("./data", download=True,
                                              transform=transforms.Compose([transforms.ToTensor()]))
test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False,
                                             transform=transforms.Compose([transforms.ToTensor()]))
train_loader = torch.utils.data.DataLoader(train_set, batch_size=100)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=100)

# train_images_list = np.load("data/train-images.npy")
# train_labels_list = np.load("data/train-labels.npy")
# data_size = np.shape(train_images_list)[0]
# shuffled_index = np.random.permutation(data_size)
# train_images_list = train_images_list[shuffled_index]
# train_labels_list = train_labels_list[shuffled_index]
#
# test_images_list = np.load("data/t10k-images.npy")
# test_labels_list = np.load("data/t10k-labels.npy")
# data_size = np.shape(test_images_list)[0]
# shuffled_index = np.random.permutation(data_size)
# test_images_list = test_images_list[shuffled_index]
# test_labels_list = test_labels_list[shuffled_index]
#
# train_images_list = image_normalize(train_images_list)
# test_images_list = image_normalize(test_images_list)

# 加载模型
model = FashionCNN()
model.to(device)
# 使用交叉熵损失函数
error = nn.CrossEntropyLoss()
# 定义学习率
learning_rate = 0.0005
# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 循环次数
num_epochs = 30
# 迭代次数
count = 0

# 是否画图
display = True

# 最小迭代次数
min_epochs = 5
# 最小的损失
min_loss = 0.1
# 最优模型
best_model = None

# Lists for visualization of loss and accuracy
loss_list = []
iteration_list = []
accuracy_list = []

# Lists for knowing classwise accuracy
predictions_list = []
labels_list = []

for epoch in range(num_epochs):
    # 　分批次加载数据集
    for train_images, train_labels in train_loader:
        # for i in range(len(train_images_list) // 100):
        # Transferring images and labels to GPU if available
        # train_images = torch.tensor(train_images_list[i * 100:(i + 1) * 100])
        # train_labels = torch.tensor(train_labels_list[i * 100:(i + 1) * 100])
        train_images, train_labels = train_images.to(device), train_labels.to(device)

        train = Variable(train_images.view(100, 1, 28, 28))
        train_labels = Variable(train_labels)

        # Forward pass
        outputs = model(train)
        loss = error(outputs, train_labels)

        # Initializing a gradient as 0 so there is no mixing of gradient among the batches
        optimizer.zero_grad()

        # Propagating the error backward
        loss.backward()

        # Optimizing the parameters
        optimizer.step()

        count += 1

        # Testing the model
        if not (count % 50):  # It's same as "if count % 50 == 0"
            total = 0
            correct = 0

            for test_images, test_labels in test_loader:
                # for j in range(len(test_images_list) // 100):
                #     test_images = torch.tensor(test_images_list[i * 100:(i + 1) * 100])
                #     test_labels = torch.tensor(test_labels_list[i * 100:(i + 1) * 100])
                test_images, test_labels = test_images.to(device), test_labels.to(device)
                labels_list.append(test_labels)

                test = Variable(test_images.view(100, 1, 28, 28))

                outputs = model(test)

                predictions = torch.max(outputs, 1)[1].to(device)
                predictions_list.append(predictions)
                correct += (predictions == test_labels).sum()

                total += len(test_labels)

            accuracy = correct * 100 / total
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)

        if not (count % 500):
            print("Iteration: {}, Loss: {}, Accuracy: {}%".format(count, loss.data, accuracy))

        # 一般可以默认验证集的损失函数值由下降转向上升（即最小值）处，模型的泛化能力最好。
        if epoch >= min_epochs and loss.data <= min_loss:
            print("update best model")
            min_loss = loss.data
            best_model = copy.deepcopy(model)

torch.save(best_model.state_dict(), "best_model.pth")

if display:
    # 画出迭代中的损失
    plt.figure(1)
    plt.plot(iteration_list, loss_list)
    plt.xlabel("No. of Iteration")
    plt.ylabel("Loss")
    plt.title("Iterations vs Loss")
    # 画出迭代中的准确度
    plt.figure(2)
    plt.plot(iteration_list, accuracy_list)
    plt.xlabel("No. of Iteration")
    plt.ylabel("Accuracy")
    plt.title("Iterations vs Accuracy")
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

    plt.show()
