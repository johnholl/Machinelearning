import gym
import tensorflow as tf
import numpy as np
import random
from PIL import Image


def preprocess(obs):
    img = Image.fromarray(obs, 'RGB')
    img = img.convert('L')
    img = img.resize((84,84))
    new_obs = np.array(img)
    return new_obs

# Testing the preprocess function
# env = gym.make('Breakout-v0')
# obs = env.reset()
# obs, reward, done, info = env.step(env.action_space.sample())
# print(obs.shape)
# new_obs = preprocess(obs)
# print(new_obs.shape)
# img = Image.fromarray(new_obs)
# img.show()

### Build the network ###

# defining helper functions


def weight_variable(shape, name):
    initial = tf.zeros(shape)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')

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
fc2_weight = weight_variable(shape=[256, 6], name='fc2_weight')
fc2_bias = bias_variable(shape=[6], name='fc2_bias')
output = tf.matmul(fc1_layer, fc2_weight) + fc2_bias

target = tf.placeholder(tf.float32, None)
action_hot = tf.placeholder('float', [None,6])
action_readout = tf.reduce_sum(tf.mul(output, action_hot), reduction_indices=1)
loss = tf.reduce_mean(tf.square(tf.sub(action_readout, target)))
train_operation = tf.train.RMSPropOptimizer(0.001, decay=0.9).minimize(loss)

weights = [conv1_weight, conv1_bias, conv2_weight, conv2_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias]

### Describe the main training loop ###

# helper function that anneals randomness over first million time steps
def update_randomness(p):
    p = .9999997*p
    if p<0.1:
        p=0.1
    return p

env = gym.make('Breakout-v0')
session = tf.Session()
session.run(tf.initialize_all_variables())
target_weights = session.run(weights)
replay_memory = []
episode_step_count = []
total_steps = 0

for episode in range(100):
    obs1 = env.reset()
    obs2 = env.step(env.action_space.sample())[0]
    obs3 = env.step(env.action_space.sample())[0]
    obs4, _, done, _ = env.step(env.action_space.sample())
    obs1, obs2, obs3, obs4 = preprocess(obs1), preprocess(obs2), preprocess(obs3), preprocess(obs4)
    state = [obs1, obs2, obs3, obs4]
    state = np.transpose(state, (1, 2, 0))
    steps = 0
    prob = 1.0

    while not done:
        if random.random() > prob:
            Q_vals = session.run(output, feed_dict={input: [state]})
            action = Q_vals.argmax()
        else:
            action = env.action_space.sample()

        prob = update_randomness(prob)
        obs1 = obs2
        obs2 = obs3
        obs3 = obs4
        obs4, reward, done, info = env.step(action)
        obs4 = preprocess(obs4)
        new_state = [obs1, obs2, obs3, obs4]
        new_state = np.transpose(new_state, (1, 2, 0))

        if done:
            reward = -1

        replay_memory.append((state, action, reward, new_state, done))
        state = new_state

        if len(replay_memory) > 10000:
            replay_memory.pop(0)

        if len(replay_memory) >= 32:
            # here is where the training procedure takes place

            # compute the one step q-values w.r.t. old weights (ie y in the loss function (y-Q(s,a,0))^2)
            # Also defines the one-hot readout action vectors
            minibatch = random.sample(replay_memory, 32)
            next_states = [m[3] for m in minibatch]
            feed_dict = {input: next_states}
            feed_dict.update(zip(weights, target_weights))
            q_vals = session.run(output, feed_dict=feed_dict)
            max_q = q_vals.max(axis=1)
            target_q = np.zeros(32)
            action_list = np.zeros((32,6))
            for i in range(32):
                _, action_index, reward, _, terminal = minibatch[i]
                target_q[i] = reward
                if not terminal:
                    target_q[i] =+ 0.99*max_q[i]

                action_list[i][action_index] = 1.0

            states = [m[0] for m in minibatch]
            feed_dict = {input: states, target: target_q, action_hot: action_list}
            session.run(train_operation, feed_dict=feed_dict)

        total_steps += 1
        steps += 1

        if done:
            break

    if total_steps > 10000000:
        break

    episode_step_count.append(steps)
    mean_steps = np.mean(episode_step_count[-100:])
    print("Training episode = {}, Total steps = {}, Last 100 mean steps = {}"
          .format(episode, total_steps, mean_steps))

    if episode % 20 == 0:
        target_weights = session.run(weights)

target_weights = session.run(weights)
np.save('weights', target_weights)




