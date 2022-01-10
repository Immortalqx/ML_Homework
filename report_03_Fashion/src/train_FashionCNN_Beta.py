# FIXME
#  这里偷了个懒，实在是不想彻彻底底从头开始训练了，所以就把之前的模型加再进来，单独针对新弄的数据集做了一次训练
#  暂时还没有好好了解这样做的后果，不过应该和正常训练差不多。
import copy

import torch
import torch.nn as nn
from torch.autograd import Variable
# from torch.utils.data import Dataset

from FashionCNN import FashionCNN
from processImage import get_dataset

# 尝试使用GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载数据集
train_images_all, train_labels_all = get_dataset("train/", True, False, 30000)
test_images_all, test_labels_all = get_dataset("test/", True, False, 3000)

# test_set = torchvision.datasets.FashionMNIST("./data", download=True, train=False,
#                                              transform=transforms.Compose([transforms.ToTensor()]))
# test_loader = torch.utils.data.DataLoader(test_set, batch_size=100)

# 加载模型
model = FashionCNN()
model.to(device)
model.load_state_dict(torch.load('best_model.pth'))
# 使用交叉熵损失函数
error = nn.CrossEntropyLoss()
# 定义学习率
learning_rate = 0.0001
# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 循环次数
num_epochs = 200
# 迭代次数
count = 0

# 最小迭代次数
min_epochs = 5
# 最小的损失
min_loss = 1.6
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
    for i in range(len(train_images_all) // 100):
        train_images = torch.tensor(train_images_all[i * 100:(i + 1) * 100])
        train_labels = torch.tensor(train_labels_all[i * 100:(i + 1) * 100])
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
        if not (count % 5):  # It's same as "if count % 50 == 0"
            total = 0
            correct = 0

            # for test_images, test_labels in test_loader:
            for j in range(len(test_images_all) // 100):
                test_images = torch.tensor(test_images_all[j * 100:(j + 1) * 100])
                test_labels = torch.tensor(test_labels_all[j * 100:(j + 1) * 100])
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

        if not (count % 10):
            print("Iteration: {}, Loss: {}, Accuracy: {}%".format(count, loss.data, accuracy))

        # 一般可以默认验证集的损失函数值由下降转向上升（即最小值）处，模型的泛化能力最好。
        if epoch >= min_epochs and loss.data <= min_loss:
            print("update best model")
            min_loss = loss.data
            best_model = copy.deepcopy(model)

torch.save(best_model.state_dict(), "new_best_model.pth")
