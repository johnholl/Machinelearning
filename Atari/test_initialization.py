import gym
import tensorflow as tf
import numpy as np
from PIL import Image


def preprocess(obs):
    img = Image.fromarray(obs, 'RGB')
    img = img.convert('L')
    img = img.resize((84,84))
    new_obs = np.array(img)
    return new_obs

def weight_variable(shape, name):
    initial = tf.random_normal(shape, stddev=0.01)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape, name=name)
    return tf.Variable(initial)



def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')


# Extract weights from npy file
weights = np.load("weights_10.npy", encoding="latin1")
print(np.shape(weights[0]))

# network

sess = tf.Session()
input = tf.placeholder(tf.float32, shape=[None, 84, 84, 4])
conv1_weight = weight_variable(shape=[8, 8, 4, 16], name='conv1_weight')
conv1_bias = bias_variable(shape=[16], name='conv1_bias')
conv1_layer = tf.nn.relu(conv2d(input, conv1_weight, 4) + conv1_bias)
conv2_weight = weight_variable(shape=[4, 4, 16, 32], name='conv2_weight')
conv2_bias = bias_variable(shape=[32], name='conv2_bias')
conv2_layer = tf.nn.relu(conv2d(conv1_layer, conv2_weight, 2) + conv2_bias)
conv2_layer_flattened = tf.reshape(conv2_layer, [-1, 9*9*32])
fc1_weight = weight_variable(shape=[9*9*32, 256], name='fc1_weight')
fc1_bias = bias_variable(shape=[256], name='fc1_bias')
fc1_layer = tf.nn.relu(tf.matmul(conv2_layer_flattened, fc1_weight) + fc1_bias)
fc2_weight = weight_variable(shape=[256, 3], name='fc2_weight')
fc2_bias = bias_variable(shape=[3], name='fc2_bias')
output = tf.matmul(fc1_layer, fc2_weight) + fc2_bias

env = gym.make('Breakout-v0')
session = tf.Session()
session.run(tf.initialize_all_variables())
replay_memory = []
episode_step_count = []
total_steps = 0

for episode in range(1):
    obs1 = env.reset()
    obs2 = env.step(env.action_space.sample())[0]
    obs3 = env.step(env.action_space.sample())[0]
    obs4, _, done, _ = env.step(env.action_space.sample())
    obs1, obs2, obs3, obs4 = preprocess(obs1), preprocess(obs2), preprocess(obs3), preprocess(obs4)
    state = np.transpose([obs1, obs2, obs3, obs4], (1, 2, 0))
    steps = 0

    while not done:
        env.render()
        Q_vals = session.run(output, feed_dict={input: [state]})
        action_index = Q_vals.argmax()
        if action_index == 0:
            action = 0
        if action_index == 1:
            action = 4
        if action_index == 2:
            action = 5
        print(action)
        obs1 = obs2
        obs2 = obs3
        obs3 = obs4
        new_obs1, reward1, _, _ = env.step(action)
        env.render()
        new_obs2, reward2, _, _ = env.step(action)
        env.render()
        new_obs3, reward3,  _, _ = env.step(action)
        env.render()
        new_obs4, reward4, done, _ = env.step(action)
        obs1 = preprocess(new_obs1)
        obs2 = preprocess(new_obs2)
        obs3 = preprocess(new_obs3)
        obs4 = preprocess(new_obs4)
        new_state = np.transpose([obs1, obs2, obs3, obs4], (1, 2, 0))
        state = new_state
        steps += 4

    print("episode ", episode, " ran for ", steps, " steps")