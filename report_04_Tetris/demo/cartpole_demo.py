import numpy as np  # 导入numpy
import gym  # 导入gym

env = gym.make('CartPole-v0').unwrapped  # 使用gym库中的环境：CartPole，且打开封装(若想了解该环境，请自行百度)

for i in range(400):  # 400个episode循环
    print('<<<<<<<<<Episode: %s' % i)
    s = env.reset()  # 重置环境
    episode_reward_sum = 0  # 初始化该循环对应的episode的总奖励

    while True:  # 开始一个episode (每一个循环代表一步)
        env.render()  # 显示实验动画
        a = np.random.randint(0, 2)  # 输入该步对应的状态s，选择动作
        s_, r, done, info = env.step(a)  # 执行动作，获得反馈

        """居然和俄罗斯方块一样！！！
        <class 'numpy.ndarray'>
        <class 'float'>
        <class 'bool'>
        <class 'dict'>
        """
        print(type(s_))
        print(type(r))
        print(type(done))
        print(type(info))

        if done:  # 如果done为True
            # round()方法返回episode_reward_sum的小数点四舍五入到2个数字
            # print('episode%last_state---reward_sum: %last_state' % (i, round(episode_reward_sum, 2)))
            break
