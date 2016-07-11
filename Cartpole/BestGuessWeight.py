import gym
import numpy as np
import tensorflow as tf

## Best results from all algorithms:
##

env = gym.make('CartPole-v0')
sess = tf.InteractiveSession()
observation = env.reset()
network_input = tf.placeholder("float", shape=(1,4))
weights = tf.constant([[-0.01729745,  0.10093228],
 [-0.00682154,  0.10542823],
 [ 0.01395494, -0.11103447],
 [ 0.01097237, -0.10515791]], shape=(4,2))
bias = tf.constant([ 0.01558696,  0.10827672], shape=(1,2))
output_vector = tf.add(tf.matmul(network_input, weights), bias)
tf.initialize_all_variables()

for i in range(1000):
	env.render()
	observation = np.reshape(observation, (1,4))
	action = np.argmax(sess.run(output_vector, feed_dict={
	network_input: observation
	}))

	observation, reward, done, info = env.step(action)
	print(reward);