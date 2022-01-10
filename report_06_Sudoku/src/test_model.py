import torch
import time
from torch.autograd import Variable
from torch.utils.data import Dataset

import SudokuCNN as cnn
import SudokuRNN as rnn
import get_data as my_load


def test_CNN_model(model_path="SudokuCNN_model_1.40394.pth", test_size=10000):
    # 公平起见，这里也要使用cpu，从而与RNN进行对比
    device = torch.device("cpu")

    # 加载数据集
    batch_size = 100
    # 这里因为是测试，所以就取分割数据中的训练集部分做测试了(不是训练模型的训练集！)
    test_set, _ = my_load.load_dataset("./data/sudoku.csv", False, test_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    # 加载模型
    model = cnn.SudokuCNN()
    model.to(device)
    model.load_state_dict(torch.load(model_path))

    # Testing the model
    total = 0
    correct = 0

    # 开始计时
    start = time.clock()

    for test_quiz, test_label in test_loader:
        test_quiz, test_label = test_quiz.to(device), test_label.to(device)
        test_quiz = Variable(test_quiz.view(test_quiz.shape[0], 1, 9, 9))
        test_label = Variable(test_label.view(-1))

        outputs = model(test_quiz)
        outputs = outputs.view(-1, 9)

        predictions = torch.max(outputs, 1)[1].to(device)
        correct += (predictions == test_label).sum()

        total += len(test_label)

    # 结束计时
    end = time.clock()

    accuracy = correct * 100 / total

    print("SudokuCNN accuracy: {}%".format(accuracy))
    print("SudokuCNN cells in error: {}".format(total - correct))
    print("SudokuCNN time = %s" % str(end - start))


def test_RNN_model(model_path="SudokuRNN_model_0.00001.pth", test_size=10000):
    # 尝试使用GPU
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")

    # 加载数据集
    batch_size = 100
    # 这里因为是测试，所以就取分割数据中的训练集部分做测试了(不是训练模型的训练集！)
    test_set, _ = my_load.load_dataset("./data/sudoku.csv", True, test_size)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size)

    # 加载模型
    model = rnn.SudokuRNN(my_load.create_constraint_mask())
    model.to(device)
    model.load_state_dict(torch.load(model_path))

    # Testing the model
    total = 0
    correct = 0

    # 开始计时
    start = time.clock()

    for test_quiz, test_label in test_loader:
        test_quiz, test_label = test_quiz.to(device), test_label.to(device)
        test_quiz = Variable(test_quiz)
        test_label = Variable(test_label)

        test_pred, test_fill = model(test_quiz)

        correct += (test_fill.max(dim=2)[1] == test_label.max(dim=2)[1]).sum().item()

        total += len(test_label) * 81

    # 结束计时
    end = time.clock()

    accuracy = correct * 100 / total

    print("SudokuRNN accuracy: {}%".format(accuracy))
    print("SudokuRNN cells in error: {}".format(total - correct))
    print("SudokuRNN time = %s" % str(end - start))


if __name__ == "__main__":
    print("===================对CNN模型进行测试===================")
    test_CNN_model()
    print("====================================================")
    print("===================对RNN模型进行测试===================")
    test_RNN_model()
    print("====================================================")
