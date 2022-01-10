import torch
import torch.utils.data as data
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt

import get_data as my_load
import SudokuRNN as my_RNN

batch_size = 100

train_set, test_set = my_load.load_dataset("./data/sudoku.csv", 50000)

dataloader_ = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

dataloader_val_ = data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

loss_function = nn.MSELoss()

sudoku_solver = my_RNN.SudokuRNN(my_load.create_constraint_mask())

optimizer = optim.Adam(sudoku_solver.parameters(), lr=0.01, weight_decay=0.000)

# 感觉效果很好，一个轮回就可以训练的很好了
epochs = 1

display = True

iteration_list = []
loss_list = []
error_list = []
accuracy_list = []

accuracy = 0
min_loss = float('inf')

for e in range(epochs):
    for i_batch, ts_ in enumerate(dataloader_):
        sudoku_solver.train()
        optimizer.zero_grad()

        pred, mat = sudoku_solver(ts_[0])

        loss = loss_function(pred, ts_[1])

        loss.backward()

        optimizer.step()

        print("Epoch " + str(e) + " batch " + str(i_batch) + ": " + str(loss.item()))

        sudoku_solver.eval()

        # with 语句适用于对资源进行访问的场合，确保不管使用过程中是否发生异常都会执行必要的“清理”操作，释放资源，
        #   比如文件使用后自动关闭／线程中锁的自动获取和释放等。
        # torch.no_grad()是一个上下文管理器，用来禁止梯度的计算，通常用来网络推断中，它可以减少计算内存的使用量。
        with torch.no_grad():
            n = 100

            # TODO 这里修改一下，改成train_CNN这样的形式！
            rows = torch.randperm(test_set.tensors[0].shape[0])[:n]
            test_pred, test_fill = sudoku_solver(test_set.tensors[0][rows])

            errors = test_fill.max(dim=2)[1] != test_set.tensors[1][rows].max(dim=2)[1]

            accuracy = (1 - errors.sum().item() / len(errors.view(-1)))
            min_loss = loss.item()

            iteration_list.append(i_batch)
            loss_list.append(loss.item())
            accuracy_list.append(accuracy)

            print("Cells in loss_fun: %d" % (errors.sum().item()))
            print("Accuracy: %lf " % accuracy)

torch.save(sudoku_solver.state_dict(), "SudokuRNN_model_%.5lf.pth" % min_loss)

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
