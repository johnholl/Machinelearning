import numpy as np
from matplotlib import pyplot as plt


data = np.load("/home/john/code/pythonfiles/Machinelearning/Atari/graph_data.npy")
time_vals = [data[i][0] for i in range(len(data))]
Q_vals = [data[i][1] for i in range(len(data))]
reward_vals = [data[i][2] for i in range(len(data))]
step_vals = [data[i][3] for i in range(len(data))]

plt.plot(time_vals, step_vals)
plt.show()