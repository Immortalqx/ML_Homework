# report 05 拼图游戏



[TOC]

# 参考资料

## 如何设计这个网络的?

在尝试了20多种神经网络架构和大量的尝试和错误之后，我得到了一个最优的设计。如下所示。

首先，从图像中提取每一块拼图(共4块)。

然后把每一个片段都传递给CNN。CNN提取有用的特征并输出一个特征向量。

我们使用Flatten layer将所有4个特征向量连接成一个。

然后我们通过前馈网络来传递这个组合向量。这个网络的最后一层给出了一个16单位长的向量。

我们将这个16单位向量重塑成4x4的矩阵。

## **为什么要做维度重塑?**

在一个正常的分类任务中，神经网络会为每个类输出一个分数。我们通过应用softmax层将该分数转换为概率。概率值最高的类就是我们预测的类。这就是我们如何进行分类。

这里的情况不同。我们想把每一个片段都分类到正确的位置(0,1,2,3)，这样的片段共有4个。所以我们需要4个向量(对于每个块)每个有4个分数(对于每个位置)，这只是一个4x4矩阵。其中的行对应于要记分的块和列。最后，我们在这个输出矩阵行上应用一个softmax。

![img](https://img-blog.csdnimg.cn/20200731084405752.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L20wXzQ2NTEwMjQ1,size_16,color_FFFFFF,t_70)

```python
model = keras.models.Sequential()

model.add(td(ZeroPadding2D(2), input_shape=(4,100,100,3)))  # extra padding

model.add(td(Conv2D(50, kernel_size=(5,5), padding='same', activation='relu', strides=2))) # padding=same for more padding
model.add(td(BatchNormalization()))
model.add(td(MaxPooling2D()))                                                              # only one maxpool layerQ

model.add(td(Conv2D(100, kernel_size=(5,5), padding='same', activation='relu', strides=2)))
model.add(td(BatchNormalization()))
model.add(td(Dropout(0.3)))

model.add(td(Conv2D(100, kernel_size=(3,3), padding='same', activation='relu', strides=2)))
model.add(td(BatchNormalization()))
model.add(td(Dropout(0.3)))

model.add(td(Conv2D(200, kernel_size=(3,3), padding='same', activation='relu', strides=1)))
model.add(td(BatchNormalization()))
model.add(td(Dropout(0.3)))

model.add(Flatten())  # combining all the features

model.add(Dense(600, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(400, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(16))
model.add(Reshape((4, 4)))        # reshaping the final output
model.add(Activation('softmax'))  # softmax would be applied row wise

```

### 模型解释
输入形状是(4,100,100,3)。我将形状(100,100,3)的4个图像(拼图)输入到网络中。

我使用的是时间分布(TD)层。TD层在输入上多次应用给定的层。在这里，TD层将对4个输入图像应用相同的卷积层(行:5,9,13,17)。

为了使用TD层，我们必须在输入中增加一个维度，TD层在该维度上多次应用给定的层。这里我们增加了一个维度，即图像的数量。因此，我们得到了4幅图像的4个特征向量。

一旦CNN特征提取完成，我们将使用Flatten层(行:21)连接所有的特征。然后通过前馈网络传递矢量。重塑最终的输出为4x4矩阵，并应用softmax(第29,30行)。

### CNN的架构
这个任务与普通的分类任务完全不同。在常规的分类中，任务网络更关注图像的中心区域。但在拼图游戏中，边缘信息比中心信息重要得多。

所以我的CNN架构与平常的CNN有以下几个不同之处。

### 填充

我在图像通过CNN之前使用了一些额外的填充(line: 3)，并且在每次卷积操作之前填充feature map (padding = same)，以保护尽可能多的边缘信息。

### MaxPooling

代码中尽量避免了pooling层，只使用一个MaxPool层来减小feature map的大小(行:7). pooling使得网络平移不变性，这意味着即使你旋转或晃动图像中的对象，网络仍然会检测到它。这对任何对象分类任务都很有用。

对于拼图游戏一般不希望网络具有平移不变性。我们的网络应该对变化很敏感。因为我们的边缘信息是非常敏感的。

### 浅层网络

我们知道CNN的顶层提取了像边缘、角等特征。当我们深入更深的层倾向于提取特征，如形状，颜色分布，等等。这和我们的案例没有太大关系，所以只创建一个浅层网络。

这些都是您需要了解CNN架构的重要细节。网络的其余部分相当简单，有3个前馈层，一个重塑层，最后一个softmax层。

### 训练
最后，我使用sparse_categorical_crossentropy loss和adam optimizer编译我的模型。我们的目标是一个4单位向量，告诉我们每一块的正确位置。
我把网络训练了5个轮次。我开始时的学习率是0.001批次大小是64。在每一个轮次之后，我都在降低学习速度，增加批处理规模。



# 拼图问题

这个真的要做的话，问题会比较多。

1. 图片本身是三通道的，对图片进行分割也是必须的，这样的话会不会冲突？？？

   比如不分割，原始输入是3\*200\*200的

   但是如果分割的话，对于2\*2拼图，原始输入就变成了4\*3\*100\*100

   这样的话是处理成12\*100\*100的好，还是想办法弄成4\*3\*100\*100

2. 