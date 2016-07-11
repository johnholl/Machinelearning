import numpy as np
import gym


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


env = gym.make('CartPole-v0')
observation = env.reset()
weights = np.random.uniform(-1,1,(4))
Reward = runExperiment(weights)

for i in range(10000):
	new_weights = weights + np.random.uniform(-0.1, 0.1, (4))
	newReward1 = runExperiment(new_weights)
	newReward2 = runExperiment(new_weights)
	newReward3 = runExperiment(new_weights)
	newReward = min(newReward1, newReward2, newReward3)
	if(newReward > Reward):
		print(newReward)
		Reward = newReward
		weights = new_weights
	if(Reward==1000):
		print(i)
		print(Reward)
		print(weights)
		break

