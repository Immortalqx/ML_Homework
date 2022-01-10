import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset

import SudokuCNN as my_model
import get_data as my_load

# 尝试使用GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载数据集
batch_size = 100
train_set, test_set = my_load.load_dataset("./data/sudoku.csv", False, 80000)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

# 加载模型
model = my_model.SudokuCNN()
model.to(device)
# 损失函数
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
min_epochs = 10
# 最小的损失
min_loss = float('inf')
# 最优模型
best_model = None

# Lists for visualization of loss_function and accuracy
loss_list = []
iteration_list = []
accuracy_list = []

for epoch in range(num_epochs):
    # 　分批次加载数据集
    for train_quiz, train_label in train_loader:
        train_quiz, train_label = train_quiz.to(device), train_label.to(device)
        train_quiz = Variable(train_quiz.view(train_quiz.shape[0], 1, 9, 9))
        train_label = Variable(train_label.view(-1))

        # Forward pass
        outputs = model(train_quiz)
        outputs = outputs.view(-1, 9)

        loss = error(outputs, train_label.long())

        # Initializing a gradient as 0 so there is no mixing of gradient among the batches
        optimizer.zero_grad()

        # Propagating the loss_fun backward
        loss.backward()

        # Optimizing the parameters
        optimizer.step()

        count += 1

        # print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch, count, loss_function.item()))

        # Testing the model
        if not (count % 500):  # It's same as "if count % 50 == 0"
            total = 0
            correct = 0

            for test_quiz, test_label in test_loader:
                test_quiz, test_label = test_quiz.to(device), test_label.to(device)
                test_quiz = Variable(test_quiz.view(test_quiz.shape[0], 1, 9, 9))
                test_label = Variable(test_label.view(-1))

                outputs = model(test_quiz)
                outputs = outputs.view(-1, 9)

                predictions = torch.max(outputs, 1)[1].to(device)
                correct += (predictions == test_label).sum()

                total += len(test_label)

            accuracy = correct * 100 / total
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)

        if not (count % 1000):
            print("Iteration: {}, Loss: {}, Accuracy: {}%".format(count, loss.data, accuracy))

        # 一般可以默认验证集的损失函数值由下降转向上升（即最小值）处，模型的泛化能力最好。
        if epoch >= min_epochs and loss.data <= min_loss:
            print("update best model")
            min_loss = loss.data
            best_model = copy.deepcopy(model)

torch.save(best_model.state_dict(), "SudokuCNN_model_%.5lf.pth" % min_loss)

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

    plt.show()