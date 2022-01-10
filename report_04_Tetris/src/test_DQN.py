import report_04_tools as tools
from Tetris_DQN import DQN, MEMORY_CAPACITY
import Tetris as Game

# Open AI gym提供了许多不同的环境。每一个环境都有一套自己的参数和方法。然而，他们通常由一个类Env包装（就像这是面向对象编程语言（OOPLs）的一个接口）。这个类暴露了任一环境的最常用的，最本质的方法，比如step
# ，reset，seed。拥有这个“接口”类非常好，因为它允许您的代码不受环境限制。如果您希望在不同的环境中测试单个代理，那么它还使事情变得更简单。然而，如果你想访问一个特定环境的场景动态后面的东西，需要使用unwrapped属性。
# 还原env的原始设置，env外包了一层防作弊层
# env = gym_tetris.make('TetrisA-v0').unwrapped
# 这里采用一种奖励分数惩罚身高的俄罗斯方块
env = Game.Tetris(10)

dqn = DQN()
dqn.load()  # 使用已有的模型继续训练
dqn.update_epsilon(1)  # 永远都使用最优选择

for i in range(50):  # 400个episode循环
    print('<<<<<<<<<Episode: %s' % i)
    last_state = env.reset()  # 重置环境

    episode_reward_sum = 0  # 初始化该循环对应的episode的总奖励
    episode_lines_cleared = 0  # 消除的行数

    while True:  # 开始一个episode (每一个循环代表一步)
        action = dqn.choose_action(last_state)  # 输入该步对应的状态s，选择动作
        state, reward, done, erase_count = env.machine_render(action)  # 执行动作，获得反馈

        # ===========================奖励修改部分===========================
        #
        # =================================================================
        # ===========================参数修改部分===========================
        #
        # =================================================================

        episode_reward_sum += reward  # 逐步加上一个episode内每个step的reward
        episode_lines_cleared += erase_count

        last_state = state  # 更新状态

        if done:  # 如果done为True
            # round()方法返回episode_reward_sum的小数点四舍五入到2个数字
            print('episode%s---reward_sum: %s' % (i, round(episode_reward_sum, 2)))
            print('episode%s---lines_cleared: %s' % (i, round(episode_lines_cleared, 2)))
            break
