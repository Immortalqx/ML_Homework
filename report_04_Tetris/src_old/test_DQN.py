import gym_tetris
from nes_py.wrappers import JoypadSpace
from gym_tetris.actions import SIMPLE_MOVEMENT

from Tetris_DQN import DQN

from src_old.report_04_tools import get_matrix

env = gym_tetris.make('TetrisA-v2').unwrapped
env = JoypadSpace(env, SIMPLE_MOVEMENT)

dqn = DQN()
dqn.load()  # 加载模型

for i in range(30):  # 演示30轮
    print('<<<<<<<<<Episode: %s' % i)
    s = env.reset()  # 重置环境
    s = get_matrix(s)
    episode_reward_sum = 0  # 初始化该循环对应的episode的总奖励

    while True:  # 开始一个episode (每一个循环代表一步)
        env.render()  # 显示实验动画
        a = dqn.choose_action(s)  # 输入该步对应的状态s，选择动作
        s_, r, done, info = env.step(a)  # 执行动作，获得反馈
        s_ = get_matrix(s_)

        episode_reward_sum += r  # 逐步加上一个episode内每个step的reward

        s = s_  # 更新状态

        if done:  # 如果done为True
            # round()方法返回episode_reward_sum的小数点四舍五入到2个数字
            print('episode%s---reward_sum: %s' % (i, round(episode_reward_sum, 2)))
            break
