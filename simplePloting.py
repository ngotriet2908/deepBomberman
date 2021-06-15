import matplotlib.pyplot as plt
import statistics
import os
import matplotlib
import numpy as np

# folderPath = "/Users/capu/Documents/DeepQLearningBomberman/simple"
folderPath = "simple"
# matplotlib.use('TkAgg')

for file in os.listdir(folderPath):
    if file.endswith(".txt"):
        f = open(os.path.join(folderPath, file))
        mean = np.array(f.readline().split("/")).astype(np.float)

        f = -1
        for i in range(0, len(mean)):
            if f == -1 and mean[i] > 400:
                f = mean[i]
            elif f > -1:
                mean[i] = f

        plt.plot(np.arange(len(mean)), mean, label=os.path.splitext(file)[0])

plt.ylabel('mean')
plt.xlabel('Gen')
plt.legend()
plt.show()