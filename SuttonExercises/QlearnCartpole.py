import numpy as np
import gym

def e_greedy_policy(obs, e):
	u = np.random.uniform(0.0,1.0)
	if(u<e):
		a = np.random.choice((0,1))
	else:
		a = np.argmax(Q[obs])

	return a



def Qlearn(Q, rate, discount, obs1, a, obs2, reward):
	Q[obs1][a] = Q[obs1][a] + rate*(reward + discount*max(Q[obs2])-Q[obs1][a])
	return Q

def Sarsa(Q, rate, discount, s1, s2, a, s1new, s2new, reward):

	a2 = e_greedy_policy(s1new,s2new, 0.1)
	Q[obs1][a] = Q[obs1][a] + rate*(reward + discount*Q[obs2][a2]-Q[obs2][a])
	return Q

Q = np.zeros((4,2))
env = gym.make('CartPole-v0')


# this controls the number of episodes
for i in range(0,1000):

	print ("step %d", i)

	done = False

	# initiates environment for new episode, gets initial obs
	obs1 = env.reset()

	# this loops at every episode
	while(not done):

		a = e_greedy_policy(obs1, 0.1)
		obs2, reward, done, info = env.step(a)
		Q = Qlearn(Q, 0.1, 1.0, obs1, a, obs2, reward)


# Q should be learned at this point. now let's test


