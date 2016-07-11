import numpy as np
import gym

#
# Created by johnholl on 6/10/16
#
#
#
#

def runExperiment(weights):
	env = gym.make('CartPole-v0')
	observation = env.reset()
	totalReward = 0
	for i in range(1000):
		weighted_sum = np.dot(observation, weights)
		if(weighted_sum>0):
			action = 1
		else:
			action = 0

		observation, reward, done, info = env.step(action)
		totalReward += reward
	return totalReward

w = np.random.uniform(-1, 1, (10000,4))

best_score = 0
best_weights = w[0]

for i in range(1000):
    totalReward = runExperiment(w[i])
    print("The total reward for experiment %d is %d" % (i, totalReward))
    if(totalReward > best_score):
        best_score = totalReward
        best_weights = w[i]
    else:
        print("oops")

print(best_score)
print("/n")
print(best_weights)
