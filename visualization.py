import numpy as np
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

itr = [500, 1000,2000,3000,4000,5000]

hm_acc_2 = [0.866,0.5,0.897,0.5,0.5,0.5]
hm_acc_3 = [0.5,0.767,0.5,0.501,0.5,0.5]
hm_acc_4 = [0.5,0.5,0.5,0.5,0.5,0.5]
hm_acc_5 = [0.5,0.9715,0.5,0.5,0.5,0.852]

rd_acc_2 = [0.9255,0.861,0.94,0.961,0.917,0.925]
rd_acc_3 = [0.6625,0.9415,0.9025,0.9195,0.8995,0.9355]
rd_acc_4 = [0.967,0.903,0.9535,0.8995,0.902,0.9325]
rd_acc_5 = [0.5,0.912,0.929,0.919,0.9115,0.941]

lin_acc = [0.5,0.9615,0.973,0.974,0.9725, 0.9775]

rd_tm = [1.08,2.18,4.33,7.20,9.45,11.36]
hm_tm = [0.38,1.22,2.53,4.0,11,13.4]
ln_tm = [0.1,0.1,0.1,0.1,0.1,0.1]

plt.plot(itr, rd_tm,'r',label='Radial basis')
plt.plot(itr, hm_tm,'g',label='Homogenious polynomial')
plt.plot(itr, ln_tm, 'b', label='Linear')
plt.xlabel("Number of Iterations")
plt.ylabel("Accuracy")
plt.title("Iterations vs Accuracy for various kernels of SVM")
plt.legend()
plt.savefig("iter_vs_acc.png")