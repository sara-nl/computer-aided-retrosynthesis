import numpy as np
import matplotlib.pyplot as plt


data = np.genfromtxt('scaling_data.csv', delimiter=',', skip_header=1)
num_nodes = data[:, 0]
num_workers = data[:, 1] / 2
comp_speedup_normal = data[:, 2]
comp_speedup_large = data[:, 3]
ideal = data[:, -1]

ideal_color = 'k'
normal_color = 'b'
large_color = 'r'

plt.figure(figsize=(12,10))
plt.loglog(num_workers, ideal, label='Ideal scaling', c=ideal_color)
plt.loglog(num_workers, comp_speedup_normal, '^-', markersize=12, label='Computational speedup factor', c=normal_color)
# plt.loglog(num_workers, comp_speedup_large, '*-', markersize=12, label='Computational speedup factor large model', c=large_color)
plt.legend(fontsize=21)
plt.xticks([2**i for i in range(0, 8)], [str(2**i) for i in range(0, 8)], fontsize=21)
plt.yticks([2**i for i in range(0, 8)], [str(2**i) for i in range(0, 8)], fontsize=21)
plt.xlabel('Number of nodes', fontsize=21)
plt.ylabel('Speedup', fontsize=21)
plt.grid(which='major')

for i,j,k in zip(num_workers,comp_speedup_normal, comp_speedup_large):
    plt.annotate(str(j),xy=(i,j+0.15*i), c=normal_color, fontsize=15)
    # plt.annotate(str(k),xy=(i,j-0.15*i), c=large_color, fontsize=15)

plt.show()
