# report 04 俄罗斯方块记录

There are two game modes define in NES Tetris, namely, A-type and B-type. A-type is the standard endurance Tetris game and B-type is an arcade style mode where the agent must clear a certain number of lines to win. There are three potential reward streams: (1) the change in score, (2) the change in number of lines cleared, and (3) a penalty for an increase in board height. The table below defines the available environments in terms of the game mode (i.e., A-type or B-type) and the rewards applied.

| Environment  | Game Mode | reward score | reward lines | penalize height |
| ------------ | --------- | ------------ | ------------ | --------------- |
| `TetrisA-v0` | A-type    | ✅            | ✕            | ✕               |
| `TetrisA-v1` | A-type    | ✕            | ✅            | ✕               |
| `TetrisA-v2` | A-type    | ✅            | ✕            | ✅               |
| `TetrisA-v3` | A-type    | ✕            | ✅            | ✅               |
| `TetrisB-v0` | B-type    | ✅            | ✕            | ✕               |
| `TetrisB-v1` | B-type    | ✕            | ✅            | ✕               |
| `TetrisB-v2` | B-type    | ✅            | ✕            | ✅               |
| `TetrisB-v3` | B-type    | ✕            | ✅            | ✅               |

## 目前的问题

目前问题比较多，有点麻烦。。。打算先这么来：

1. 把目前能写的实验报告写好 
2. 优先尝试把基于规则的算法实现出来 
3. 想办法设计一个非常合理的奖励函数 
4. 利用基于规则的算法给DQN创造足够的数据，然后再开始学习

## 奖励函数相关

一篇有意思的文章：https://zhuanlan.zhihu.com/p/97032357



奖励函数的设计需要保证：

- 加入了合适的先验，良好的定义了问题和在一切可能状态下的对应动作。坑爹的是模型很多时候会找到作弊的手段。Alex 举的一个例子是有一个任务需要把红色的乐高积木放到蓝色的乐高积木上面，奖励函数的值基于红色乐高积木底部的高度而定。结果一个模型直接把红色乐高积木翻了一个底朝天。仔啊，你咋学坏了，阿爸对你很失望啊。
- 奖励函数的值太过稀疏。换言之大部分情况下奖励函数在一个状态返回的值都是 0。这就和我们人学习也需要鼓励，学太久都没什么回报就容易气馁。都说 21 世纪是生物的世纪，怎么我还没感觉到呢？21 世纪才刚开始呢。我等不到了啊啊啊啊啊。
- 有的时候在奖励函数上下太多功夫会引入新的偏见（bias）。
- 要找到一个大家都使用而又具有好的性质的奖励函数。这里Alex没很深入地讨论，但链接了一篇陶神（Terence Tao）的博客，大家有兴趣可以去看下。

