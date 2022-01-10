# Result

## circles

**终端输出**

```
MyNN 训练用时:	0.728545 s
MyNN 对训练集做预测:	 1.0
MyNN 对测试集做预测:	 1.0
sklearn 训练用时:	2.704032 s
sklearn 对训练集做预测:	 1.0
sklearn 对测试集做预测:	 1.0
```

**MyNN 训练集测试结果**

<img src="./images/MyNN_circles_train.png" alt="MyNN_circles_train" style="zoom: 67%;" />

**MyNN 训练集测试结果**

<img src="./images/MyNN_circles_test.png" style="zoom:67%;" />

**sklearn 训练集测试结果**

<img src="./images/sklearn_circles_train.png" style="zoom:67%;" />

**sklearn 测试集测试结果**

<img src="./images/sklearn_circles_test.png" style="zoom:67%;" />

## moons

**终端输出**

```
MyNN 训练用时:	0.660563 s
MyNN 对训练集做预测:	 0.9706666666666667
MyNN 对测试集做预测:	 0.96
sklearn 训练用时:	9.209319 s
sklearn 对训练集做预测:	 0.9746666666666667
sklearn 对测试集做预测:	 0.952
```

**MyNN 训练集测试结果**

<img src="./images/MyNN_moons_train.png" alt="MyNN_moons_train" style="zoom: 67%;" />

**MyNN 训练集测试结果**

<img src="./images/MyNN_moons_test.png" style="zoom:67%;" />

**sklearn 训练集测试结果**

<img src="./images/sklearn_moons_train.png" style="zoom:67%;" />

**sklearn 测试集测试结果**

<img src="./images/sklearn_moons_test.png" style="zoom:67%;" />

## digits

**终端输出**

```
Loss:5661.218912 accuracy0.100223
Loss:1332.399971 accuracy0.100223
Loss:1218.177300 accuracy0.102450
............
Loss:32.263688 accuracy0.997030
Loss:32.210868 accuracy0.997030
Loss:32.158215 accuracy0.997030
# 这个时间是有问题的！
MyNN 训练用时:	106.417476 s
MyNN 对训练集做预测:	 0.9970304380103935
MyNN 对测试集做预测:	 0.9666666666666667
sklearn 训练用时:	9.181445 s
sklearn 对训练集做预测:	 1.0
sklearn 对测试集做预测:	 0.9511111111111111
```

**MyNN 测试集测试结果**

右上角为预测概率，右下角为预测结果

<img src="./images/MyNN_digits_test.png" alt="MyNN_digits_test" style="zoom:67%;" />

混淆矩阵

<img src="./images/MyNN_digits_confusion_matrix.png" alt="MyNN_digits_confusion_matrix" style="zoom:67%;" />

**sklearn 测试集测试结果**

<img src="./images/sklearn_digits_test.png" alt="sklearn_digits_test" style="zoom:67%;" />

混淆矩阵

<img src="./images/sklearn_digits_confusion_matrix.png" alt="sklearn_digits_confusion_matrix" style="zoom:67%;" />