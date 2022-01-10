import gym_tetris
from nes_py.wrappers import JoypadSpace
from gym_tetris.actions import SIMPLE_MOVEMENT

import report_04_tools as tools
from Tetris_DQN import DQN, MEMORY_CAPACITY

# Open AI gym提供了许多不同的环境。每一个环境都有一套自己的参数和方法。然而，他们通常由一个类Env包装（就像这是面向对象编程语言（OOPLs）的一个接口）。这个类暴露了任一环境的最常用的，最本质的方法，比如step
# ，reset，seed。拥有这个“接口”类非常好，因为它允许您的代码不受环境限制。如果您希望在不同的环境中测试单个代理，那么它还使事情变得更简单。然而，如果你想访问一个特定环境的场景动态后面的东西，需要使用unwrapped属性。
# 还原env的原始设置，env外包了一层防作弊层
# env = gym_tetris.make('TetrisA-v0').unwrapped
# 这里采用一种奖励分数惩罚身高的俄罗斯方块
env = gym_tetris.make('TetrisA-v1').unwrapped
env = JoypadSpace(env, SIMPLE_MOVEMENT)

dqn = DQN()
dqn.load()  # 使用已有的模型继续训练

loss = float('inf')
loss_list = open('./data/loss_list.dat', 'w')
reward_sum = open('./data/reward_sum.dat', 'w')

for i in range(300):  # 400个episode循环
    print('<<<<<<<<<Episode: %s' % i)
    last_state = env.reset()  # 重置环境
    last_state = tools.get_matrix(last_state)
    last_score = 0

    episode_reward_sum = 0  # 初始化该循环对应的episode的总奖励

    if (i + 1) % 20 == 0 and i > 30:  # 防止i=0，还没开始训练的时候就保存模型
        print("save model")
        dqn.save()

    while True:  # 开始一个episode (每一个循环代表一步)
        # if i % 100 == 0:
        #     env.render()  # 显示实验动画
        env.render()  # 显示实验动画
        action = dqn.choose_action(last_state)  # 输入该步对应的状态s，选择动作
        state, reward, done, info = env.step(action)  # 执行动作，获得反馈
        state = tools.get_matrix(state)

        # ===========================奖励修改部分===========================
        score = tools.score_policy_A(state)
        # reward = reward * 2  # 奖励消除行（这里可能还是需要修改！）
        # if score - last_score != 0:
        #     print(reward, score - last_score)
        # reward += score - last_score  # 这样对reward进行修改才是正确的！！！
        if reward != 0:
            print("消除了一行，得到：{} 分".format(reward))
        reward = reward * 5 + (score - last_score)  # 原始分数(消除的行数)+新分数
        last_score = score
        # =================================================================

        # ===========================参数修改部分===========================
        #
        # =================================================================
        dqn.store_transition(last_state, action, reward, state)  # 存储样本
        episode_reward_sum += reward  # 逐步加上一个episode内每个step的reward

        last_state = state  # 更新状态
        last_action = action  # 更新动作

        if dqn.memory_counter > MEMORY_CAPACITY:  # 如果累计的transition数量超过了记忆库的固定容量2000
            # 开始学习 (抽取记忆，即32个transition，并对评估网络参数进行更新，并在开始学习后每隔100次将评估网络的参数赋给目标网络)
            loss = dqn.learn()

        if done:  # 如果done为True
            # round()方法返回episode_reward_sum的小数点四舍五入到2个数字
            print('episode%s---reward_sum: %s' % (i, round(episode_reward_sum, 2)))
            print('episode%s---loss: %s' % (i, loss))

            loss_list.write(str(loss) + "\n")
            reward_sum.write(str(episode_reward_sum) + "\n")

            break

loss_list.close()
reward_sum.close()
