import report_04_tools as tools
from Tetris_DQN import DQN, MEMORY_CAPACITY
import Tetris as Game

env = Game.Tetris(1)
dqn = DQN()
# dqn.load()

epoch = 3000
loss = float('inf')
loss_list = open('./data/loss_list.dat', 'w')
reward_sum = open('./data/reward_sum.dat', 'w')
lines_cleared = open('./data/lines_cleared.dat', 'w')

for i in range(epoch):
    dqn.update_epsilon(2 * (i - 1000) / epoch)  # 逐步的增大epsilon

    print('<<<<<<<<<Episode: %s' % i)
    last_state = env.reset()  # 重置环境
    last_score = 0

    episode_reward_sum = 0  # 初始化该循环对应的episode的总奖励
    episode_lines_cleared = 0  # 消除的行数

    if i % 100 == 0 and i > 300:  # 防止i=0，还没开始训练的时候就保存模型
        print("save model")
        dqn.save()

    while True:  # 开始一个episode (每一个循环代表一步)
        action = dqn.choose_action(last_state)  # 输入该步对应的状态s，选择动作
        # state, _, done, erase_count = env.machine_render(action)  # 执行动作，获得反馈
        state, reward, done, erase_count = env.machine_render(action)  # 执行动作，获得反馈

        # ===========================奖励修改部分===========================
        # score = tools.score_policy_A(state)
        # reward = (score - last_score)  # 原始分数(消除的行数)+新分数
        # last_score = score
        # =================================================================

        # ===========================参数修改部分===========================
        #
        # =================================================================

        dqn.store_transition(last_state, action, reward, state)  # 存储样本
        episode_reward_sum += reward  # 逐步加上一个episode内每个step的reward
        episode_lines_cleared += erase_count

        last_state = state  # 更新状态
        # last_action = action  # 更新动作

        if dqn.memory_counter > MEMORY_CAPACITY:  # 如果累计的transition数量超过了记忆库的固定容量2000
            # 开始学习 (抽取记忆，即32个transition，并对评估网络参数进行更新，并在开始学习后每隔100次将评估网络的参数赋给目标网络)
            loss = dqn.learn()

        if done:  # 如果done为True
            # round()方法返回episode_reward_sum的小数点四舍五入到2个数字
            print('episode%s---reward_sum: %s' % (i, round(episode_reward_sum, 2)))
            print('episode%s---lines_cleared: %s' % (i, round(episode_lines_cleared, 2)))
            print('episode%s---loss: %s' % (i, loss))

            loss_list.write(str(loss) + "\n")
            reward_sum.write(str(episode_reward_sum) + "\n")
            lines_cleared.write(str(episode_lines_cleared) + "\n")

            break

loss_list.close()
reward_sum.close()
lines_cleared.close()
