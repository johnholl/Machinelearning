from numpy import random
from numpy.random import normal, uniform, choice
import matplotlib.pyplot as plt
import numpy as np


# a can be 0, 1, 2, or 3. L R U D respectively. s1 is row, s2 column.


def step(s1, s2, a):
	end = False

	if(a==0):
		if(s2!=0):
			s2 = s2 - 1
		
		reward = -1.0


	if(a==1):
		if((s1==0)&(s2==0)):
			reward = -100.0

		elif((s1!=0)&(s2!=11)):
			s2 = s2 + 1
			reward = -1.0

		else:
			reward = -1.0

	if(a==2):
		if(s1!=3):
			s1 = s1 + 1

		reward = -1.0

	if(a==3):
		if((s1==1)&(s2!=0)&(s2!=11)):
			s1 = 0
			s2 = 0
			reward = -100.0

		elif((s1==1)&(s2==11)):
			s1 = s1 - 1
			reward= 0.0
			end = True

		elif(s1==0):
			reward = -1.0

		else:
			s1 = s1 - 1
			reward = -1.0

	return s1, s2, reward, end




def e_greedy_policy(s1, s2, e):
	u = np.random.uniform(0.0,1.0)
	if(u<e):
		a = np.random.choice((0,1,2,3))
	else:
		a = np.argmax(Q[s1][s2])

	return a



def Qlearn(Q, rate, discount, s1, s2, a, s1new, s2new, reward):
	Q[s1][s2][a] = Q[s1][s2][a] + rate*(reward + discount*max(Q[s1new][s2new])-Q[s1][s2][a])
	return Q

def Sarsa(Q, rate, discount, s1, s2, a, s1new, s2new, reward):

	a2 = e_greedy_policy(s1new,s2new, 0.1)
	Q[s1][s2][a] = Q[s1][s2][a] + rate*(reward + discount*Q[s1new][s2new][a2]-Q[s1][s2][a])
	return Q



#Q = np.random.uniform(-1.0,0.0,(4,12,4))
Q = np.zeros((4,12,4))

for i in range(0,1000):

	s1, s2 = 0, 0
	end = False
	reward = 0.0

	while(not end):
		a = e_greedy_policy(s1, s2, 0.1)
		s1new, s2new, reward, end = step(s1, s2, a)
		
		#Q = Qlearn(Q, 0.1, 1.0, s1, s2, a, s1new, s2new, reward)
		Q = Sarsa(Q, 0.1, 1.0, s1, s2, a, s1new, s2new, reward)
		s1 = s1new
		s2 = s2new


Q_choices = np.ndarray.argmax(Q, 2)
print Q_choices










