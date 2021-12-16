import numpy as np
import matplotlib.pyplot as plt
import matplotlib

node4 = np.genfromtxt('4node.csv', delimiter=',')
node8 = np.genfromtxt('8node.csv', delimiter=',')
node16 = np.genfromtxt('16node.csv', delimiter=',')
node32 = np.genfromtxt('32node.csv', delimiter=',')

node4[:, 0] -= node4[0, 0]
node8[:, 0] -= node8[0, 0]
node16[:, 0] -= node16[0, 0]
node32[:, 0] -= node32[0, 0]

matplotlib.rc('xtick', labelsize=21)
matplotlib.rc('ytick', labelsize=21)

plt.figure(figsize=(12, 10))
plt.plot(node4[:, 0] / 3600, 100 * node4[:, 1], label='8 workers')
plt.plot(node8[:, 0] / 3600, 100 * node8[:, 1], label='16 workers')
plt.plot(node16[:, 0] / 3600, 100 * node16[:, 1], label='32 workers')
plt.plot(node32[:, 0] / 3600, 100 * node32[:, 1], label='64 workers')
plt.ylim([85,100])
plt.yticks(np.arange(85, 101, 1))
plt.legend(fontsize=21)
plt.grid(which='major')
plt.xlabel('Training time (hours)', fontsize=21)
plt.ylabel('Validation accuracy (%)', fontsize=21)
plt.show()
