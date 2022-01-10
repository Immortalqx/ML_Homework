import numpy as np
from nes_py.wrappers import JoypadSpace
import gym_tetris
from gym_tetris.actions import SIMPLE_MOVEMENT
from src.report_04_tools import get_matrix
import time

"""
在开始之前，需要把这个俄罗斯方块的输入输出搞清楚

一部分相关信息：
state type: <class 'numpy.ndarray'>		 state shape: (240, 256, 3) 这个是当下的情况，所以应该是240*256的像素点！然后有三个通道！
reward type: <class 'numpy.int64'>		 reward shape: () 这个是奖励的数值
done type: <class 'bool'>		 done shape: () 这个应该不太重要
info type: <class 'dict'>		 info shape: () 这个是个比较麻烦的东西，我是不是需要把这个给输出出来。。。
action type: <class 'int'>		 action shape: () 这个是操作类型，所以最后输出一个int就可以了

关于state：
这个应该是学习的重点对象，这里的话，这个很明显就是一张RGB图像。
由于是俄罗斯方块，颜色不重要，可以直接调整为二值图；同时因为好几个像素点才会表示一个方格，所以可以考虑加一个比较合适的卷积层！

比较关键的dic：
{'current_piece': 'Jd', 
'number_of_lines': 0, 
'score': 0, 
'next_piece': 'Zh', 
'statistics': {'T': 0, 'J': 1, 'Z': 0, 'O': 0, 'S': 0, 'L': 0, 'I': 0}, 
'board_height': 0}
这个dic怎么使用呢？？？肯定是要用的，重点是current_piece，不过这个为啥没有表示旋转的？


比较关键的action，这里就是0，1，2，3，4，5，。。。。。。。
MOVEMENT = [
    ['NOOP'],
    ['A'],
    ['B'],
    ['right'],
    ['right', 'A'],
    ['right', 'B'],
    ['left'],
    ['left', 'A'],
    ['left', 'B'],
    ['down'],
    ['down', 'A'],
    ['down', 'B'],
]

SIMPLE_MOVEMENT = [
    ['NOOP'],
    ['A'],
    ['B'],
    ['right'],
    ['left'],
    ['down'],
]
"""


def info_out(state, reward, done, info, action):
    print("============================================================================")
    print("=========================================")
    print("state type: {}\t\t state shape: {}".format(type(state), np.shape(state)))
    print(state)
    print("=========================================")
    print("reward type: {}\t\t reward shape: {}".format(type(reward), np.shape(reward)))
    print(reward)
    print("=========================================")
    print("done type: {}\t\t done shape: {}".format(type(done), np.shape(done)))
    print(done)
    print("=========================================")
    print("info type: {}\t\t info shape: {}".format(type(info), np.shape(info)))
    print(info)
    print("=========================================")
    print("action type: {}\t\t action shape: {}".format(type(action), np.shape(action)))
    print(action)
    print("=========================================")
    print("============================================================================")


def test_Tetris(info_print=False):
    env = gym_tetris.make('TetrisA-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    done = True
    for step in range(50000):
        if done:
            state = env.reset()
        action = env.action_space.sample()
        state, reward, done, info = env.step(action)
        env.render()

        if info_print:
            get_matrix(state, True)
            info_out(state, reward, done, info, action)
        print(SIMPLE_MOVEMENT[action])
        time.sleep(2)
    env.close()


if __name__ == "__main__":
    # test_Tetris(False)
    # print(np.random.randint(0, 10))
    print(time.clock_gettime(1))

"""
SIMPLE_MOVEMENT = [
    ['NOOP'],
    ['A'],
    ['B'],
    ['right'],
    ['left'],
    ['down'],
]

"""
