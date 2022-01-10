import copy
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset

import SudokuRNN as my_model
import get_data as my_load

# FIXME 这里RNN不能使用GPU，尝试着用就会报错。感觉是定义模型的时候出的问题。。。
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

# 加载数据集
batch_size = 100
train_set, test_set = my_load.load_dataset("./data/sudoku.csv", True, 10000)
# train_set, test_set = my_load.load_dataset("./data/sudoku_test.csv", True, 1000)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

# 加载模型
model = my_model.SudokuRNN(my_load.create_constraint_mask())
model.to(device)
# 损失函数
loss_fun = nn.MSELoss()
# 定义学习率
learning_rate = 0.01
# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.000)
# 循环次数（用cpu训练，不敢弄太大了，训练时间太长）
num_epochs = 1
# 迭代次数
count = 0
# 是否画图
display = True
# 最小迭代次数
min_epochs = 0
# 最小的损失
min_loss = float('inf')
# 最优模型
best_model = None

# Lists for visualization of loss_function and accuracy
loss_list = []
iteration_list = []
accuracy_list = []
accuracy = 0

for epoch in range(num_epochs):
    # 　分批次加载数据集
    for train_quiz, train_label in train_loader:
        train_quiz, train_label = train_quiz.to(device), train_label.to(device)
        train_quiz = Variable(train_quiz)
        train_label = Variable(train_label)

        model.train()
        optimizer.zero_grad()

        pred, mat = model(train_quiz)

        loss = loss_fun(pred, train_label)

        loss.backward()

        # Optimizing the parameters
        optimizer.step()

        count += 1

        model.eval()
        # Testing the model
        if not (count % 2):  # It's same as "if count % 50 == 0"
            total = 0
            correct = 0

            for test_quiz, test_label in test_loader:
                test_quiz, test_label = test_quiz.to(device), test_label.to(device)
                test_quiz = Variable(test_quiz)
                test_label = Variable(test_label)

                test_pred, test_fill = model(test_quiz)

                correct += (test_fill.max(dim=2)[1] == test_label.max(dim=2)[1]).sum().item()

                total += len(test_label) * 81

            accuracy = correct / total
            loss_list.append(loss.data)
            iteration_list.append(count)
            accuracy_list.append(accuracy)

            print("accuracy: ", accuracy)
            print("cells in error: ", total - correct)

        if not (count % 5):
            print("Iteration: {}, Loss: {}, Accuracy: {}".format(count, loss.item(), accuracy))

        # 一般可以默认验证集的损失函数值由下降转向上升（即最小值）处，模型的泛化能力最好。
        if epoch >= min_epochs and loss.item() <= min_loss:
            # print("update best model")
            min_loss = loss.item()
            best_model = copy.deepcopy(model)

torch.save(best_model.state_dict(), "SudokuRNN_model_%.5lf.pth" % min_loss)

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
