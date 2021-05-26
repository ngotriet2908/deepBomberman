import matplotlib.pyplot as plt
import numpy as np

a = [0, 1, 2, 3, 4]
b = [2, 6, 3, 7, 9, 4, 5]
c = [1, 4, 7, 2, 6, 5, 6, 7]

fig = plt.figure()
ax = fig.add_subplot(111)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.plot(np.arange(len(a)), a)
plt.show()
plt.plot(np.arange(len(b)), b)
plt.show()
plt.plot(np.arange(len(c)), c)
plt.show()