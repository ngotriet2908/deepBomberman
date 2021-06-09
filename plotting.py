import matplotlib.pyplot as plt
import statistics

import numpy as np

file = "/Users/capu/Desktop/Bomberman/"
filename = "result.txt"

gen = 20
train_eps = 5000
eval_eps = 100

f = open(file + filename, "r")

agents = []
data = []


for x in f:
    y = x.split("/")
    # print(len(y))
    data.append(y[1:])
    agents.append(y[0])
    # print(y[:1])

plot_data_eval = []
plot_data_train = []

for x in data:
    training_data = []
    eval_data = []

    for i in range(0, len(x), (train_eps + eval_eps)):
        gen_data = x[i: i + (train_eps + eval_eps)]
        training_data.append(gen_data[:train_eps])
        eval_data.append(gen_data[train_eps: (train_eps + eval_eps)])

    plot_data_eval.append(eval_data)
    plot_data_train.append(training_data)
# print(len(training_data))
# print(len(eval_data))
# print()

# training_mean = []
# training_std = []
# for x in training_data:
#     training_mean.append(statistics.mean(np.array(x).astype(np.float)))
#     training_std.append(statistics.stdev(np.array(x).astype(np.float)))
#     # print(len(x))

fig = plt.figure()


for i in range(len(plot_data_eval)):
    eval_mean = []
    eval_std = []

    for y in plot_data_eval[i]:
        eval_mean.append(statistics.mean(np.array(y).astype(np.float)))
        eval_std.append(statistics.stdev(np.array(y).astype(np.float)))

    print(len(eval_mean))
    plt.plot(np.arange(len(eval_mean)), eval_mean, label=agents[i])

# print(len(eval_mean))

# plt.plot(np.arange(len(training_mean)), training_mean)
# plt.errorbar(
#     np.arange(len(eval_mean)),
#     eval_mean,
#     eval_std
# )
plt.ylabel('mean')
plt.xlabel('Gen')
plt.legend()
plt.show()
