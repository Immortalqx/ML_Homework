import matplotlib.pyplot as plt

loss_file = open('./src/data/round2/loss_list.dat', 'r')
reward_file = open('./src/data/round2/reward_sum.dat', 'r')
lines_file = open('./src/data/round2/lines_cleared.dat', 'r')

loss_list = []
reward_list = []
lines_list = []

for loss in loss_file:
    loss_list.append(float(loss))

for reward in reward_file:
    reward_list.append(float(reward))

for lines in lines_file:
    lines_list.append(float(lines))

plt.figure(1)
plt.plot(list(range(len(loss_list))), loss_list)
plt.xlabel("No. of Iteration")
plt.ylabel("Loss")
plt.title("Iterations vs Loss")

plt.figure(2)
plt.plot(list(range(len(reward_list))), reward_list)
plt.xlabel("No. of Iteration")
plt.ylabel("Reward")
plt.title("Iterations vs Reward")

plt.figure(3)
plt.plot(list(range(len(lines_list))), lines_list)
plt.xlabel("No. of Iteration")
plt.ylabel("Lines Cleared")
plt.title("Iterations vs Lines Cleared")

plt.show()
