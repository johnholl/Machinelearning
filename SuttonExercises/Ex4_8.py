
import numpy as np
from numpy.random import choice

## Initialize value function,action, dynamics
## state, and dummy state spaces

S = range(1,100)
S_plus = range(0,101)
A = [[0]]
for s in S:
	A.append(range(1,min(s, 100-s)+1))

V = np.zeros(101)
P = np.zeros((100,100,100))
R = np.zeros((101,101,101))

for s1 in S:
	for a in A[s1]:
		for s2 in S_plus:
			#if((s1+a==s2) or (s1-a==s2)):
			#	P[s1, a, s2] = .5

			if(s2 == 100):
				R[s1, a, s2] = 1.0

			if(s1 == 100):
				R[s1, a, s2] = 0.0


## main program loop. NOTE: only works since values always increase
for i in range(100):
	delta = 0
	for s1 in S:
		v = V[s1]
		updates = [0]*101
		updates_new = [0]*101
		for a in A[s1]:
			updates_new[s1] = .4*(R[s1][a][s1+a] + V[s1+a])+ .6*(R[s1][a][s1-a] + V[s1-a])
			updates[s1] = max(updates_new[s1], updates[s1])

		V[s1] = updates[s1]

		delta = max(delta, abs(v-V[s1]))
	print delta

	#if(delta < .000000000000000000001):
	#	break


pi = [0]
for s1 in S:
	Q_values = [0]
	for a in A[s1]:
		Q_values.append(.4*(R[s1][a][s1+a] + V[s1+a])+ .6*(R[s1][a][s1-a] + V[s1-a]))
	if(s1 ==16):
		print Q_values



	pi.append(np.argmax(Q_values))







