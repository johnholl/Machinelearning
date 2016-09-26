import gym
import tensorflow as tf
import numpy as np
from PIL import Image
import random


def preprocess(obs):
    img = Image.fromarray(obs, 'RGB')
    img = img.convert('L')
    img = img.resize((84,84))
    new_obs = np.array(img)
    return new_obs

def weight_variable(shape, name, weight_value):
    return tf.Variable(weight_value, name=name)


def bias_variable(shape, name, bias_value):
    initial = tf.constant(bias_value, shape=shape, name=name)
    return tf.Variable(initial)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')


# Extract weights from npy file
weights = np.load("weights_7.npy", encoding="latin1")
print(weights[0])

# network

sess = tf.Session()
input = tf.placeholder(tf.float32, shape=[None, 84, 84, 4])
conv1_weight = weight_variable(shape=[8, 8, 4, 16], name='conv1_weight', weight_value=weights[0])
conv1_bias = bias_variable(shape=[16], name='conv1_bias', bias_value=weights[1])
conv1_layer = tf.nn.relu(conv2d(input, conv1_weight, 4) + conv1_bias)
conv2_weight = weight_variable(shape=[4, 4, 16, 32], name='conv2_weight', weight_value=weights[2])
conv2_bias = bias_variable(shape=[32], name='conv2_bias', bias_value=weights[3])
conv2_layer = tf.nn.relu(conv2d(conv1_layer, conv2_weight, 2) + conv2_bias)
conv2_layer_flattened = tf.reshape(conv2_layer, [-1, 9*9*32])
fc1_weight = weight_variable(shape=[9*9*32, 256], name='fc1_weight', weight_value=weights[4])
fc1_bias = bias_variable(shape=[256], name='fc1_bias', bias_value=weights[5])
fc1_layer = tf.nn.relu(tf.matmul(conv2_layer_flattened, fc1_weight) + fc1_bias)
fc2_weight = weight_variable(shape=[256, 6], name='fc2_weight', weight_value=weights[6])
fc2_bias = bias_variable(shape=[6], name='fc2_bias', bias_value=weights[7])
output = tf.matmul(fc1_layer, fc2_weight) + fc2_bias

env = gym.make('Breakout-4skips-v0')
session = tf.Session()
session.run(tf.initialize_all_variables())
replay_memory = []
episode_step_count = []

for episode in range(1):
    obs1 = env.reset()
    obs2 = env.step(env.action_space.sample())[0]
    obs3 = env.step(env.action_space.sample())[0]
    obs4, _, done, _ = env.step(env.action_space.sample())
    obs1, obs2, obs3, obs4 = preprocess(obs1), preprocess(obs2), preprocess(obs3), preprocess(obs4)
    state = np.transpose([obs1, obs2, obs3, obs4], (1, 2, 0))
    steps = 0
    episode_reward = 0

    while not done:
        env.render()
        Q_vals = session.run(output, feed_dict={input: [state]})
        if random.random() > 0.05:
            action = Q_vals.argmax()
        else:
            action = random.choice([0, 1, 2, 3, 4, 5])
        obs1 = obs2
        obs2 = obs3
        obs3 = obs4
        new_obs, reward, done, _ = env.step(action)
        episode_reward += reward*(.99)**steps
        obs4 = preprocess(new_obs)
        new_state = np.transpose([obs1, obs2, obs3, obs4], (1, 2, 0))
        state = new_state
        steps += 1

    print("episode ", episode, " ran for ", steps, " steps")
    print(episode_reward)