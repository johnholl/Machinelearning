import numpy as np
import gym

env = gym.make('CartPole-v0')
env.reset()
for i in range(1000):
	env.render()
	env.step(0)