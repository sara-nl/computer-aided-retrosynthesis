import numpy as np
import matplotlib.pyplot as plt
import matplotlib

worker1 = np.genfromtxt('1worker.csv', delimiter=',')
worker4 = np.genfromtxt('4worker.csv', delimiter=',')
worker8 = np.genfromtxt('8worker.csv', delimiter=',')
worker64 = np.genfromtxt('64worker.csv', delimiter=',')
worker128 = np.genfromtxt('128worker.csv', delimiter=',')

worker64[:, 0] = worker64[:, 0] / (45033 / (64 * 32))
worker128[:, 0] = worker128[:, 0] / (45033 / (128 * 32))

worker1[:, 1] *= 100
worker4[:, 1] *= 100
worker8[:, 1] *= 100
worker64[:, 1] *= 100
worker128[:, 1] *= 100

matplotlib.rc('xtick', labelsize=21) 
matplotlib.rc('ytick', labelsize=21) 

plt.figure(figsize=(20, 7))
plt.plot(worker1[:, 0], worker1[:, 1], label='Baseline')
plt.plot(worker4[:, 0], worker4[:, 1], label='4 nodes')
plt.plot(worker8[:, 0], worker8[:, 1], label='8 nodes')
plt.plot(worker64[:, 0], worker64[:, 1], label='64 nodes')
plt.plot(worker128[:, 0], worker128[:, 1], label='128 nodes')
plt.legend(fontsize=21)
plt.xticks(np.arange(0, 2001, 200))
plt.grid(which='major')
plt.xlabel('Epochs', fontsize=21)
plt.ylabel('Learning rate * 100', fontsize=21)
plt.savefig('lr.png')
