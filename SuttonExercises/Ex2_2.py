

from numpy import random
from numpy.random import normal, uniform, choice
import matplotlib.pyplot as plt
import numpy as np




def means(arms):
	a = normal(0,1, arms)
	return a


def epsilon_greedy_policy_update(arm, Q, time, reward):
	Q[arm] = Q[arm] + (1.0/(time + 1.0))*(reward - Q[arm])
	guess_param = uniform(0.0, 1.0)
	if(guess_param < 0.2):
		selection = choice([0,1,2,3,4,5,6,7,8,9])
	else:
		selection = Q.index(max(Q))

	return selection

def softmax_policy_update()


def play(means, selection):
	reward = normal(means[selection], 1)
	return reward


running_total = [0.0]*1000

for i in range(2000):

	arms = 10
	arm_means = means(arms)
	optimal_action = np.argmax(arm_means)
	Q = [0.0]*10
	counter = [0.0]*10
	reward = 0.0
	selection = choice([0,1,2,3,4,5,6,7,8,9])
	


	for t in range(1000):
		if(selection == optimal_action):
			running_total[t] = running_total[t] + 1.0

		reward = play(arm_means, selection)
		counter[selection] = counter[selection]+1.0
		new_selection = epsilon_greedy_policy_update(selection, Q, counter[selection], reward)
		selection = new_selection

percent_correct = np.divide(running_total, 2000.0)

x = np.arange(0,1000,1)
plt.plot(x, percent_correct)
plt.show()








