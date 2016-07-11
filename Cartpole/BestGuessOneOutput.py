import gym
import numpy as np




env = gym.make('CartPole-v0')
observation = env.reset()
weights = np.array([-0.03204281,  0.26688842,  0.30378541,  0.21247193])
while 1:
	env.render()
	weighted_sum = np.dot(observation, weights)
	if(weighted_sum>0):
		action = 1
	else:
		action = 0

	observation, reward, done, info = env.step(action)
	print(action)

	if done:
		break
