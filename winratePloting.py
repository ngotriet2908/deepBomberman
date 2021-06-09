import matplotlib.pyplot as plt
import statistics
import matplotlib
import os

import numpy as np
matplotlib.use('TkAgg')
# folderPath = "/Users/capu/Documents/DeepQLearningBomberman/winrate"
folderPath = "winrate"

for file in os.listdir(folderPath):
    if file.endswith(".txt"):
        f = open(os.path.join(folderPath, file))
        f.readline()
        mean = np.array(f.readline().split("/")).astype(np.float)

        f = -1
        for i in range(0, len(mean)):
            if f == -1 and mean[i] > 400:
                f = mean[i]
            elif f > -1:
                mean[i] = f

        plt.plot(np.arange(len(mean)), mean, label=os.path.splitext(file)[0])

plt.ylabel('winrate')
plt.xlabel('Gen')
plt.legend()
plt.show()