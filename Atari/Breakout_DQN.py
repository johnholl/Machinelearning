import gym
import tensorflow as tf
import numpy as np
import random
from PIL import Image
import argparse


MAX_STEPS = 10000
#actions are 0, 4, and 5

parser = argparse.ArgumentParser(description="A DQN for playing breakout")
parser.add_argument("--initialized_weights")
args, leftovers = parser.parse_known_args()
if args.initialized_weights is not None:
    print("Loading weights at path: " + args.initialized_weights)
    weight_vector = np.load(args.initialized_weights, encoding='latin1')
    initial_w1 = weight_vector[0]
    initial_b1 = weight_vector[1]
    initial_w2 = weight_vector[2]
    initial_b2 = weight_vector[3]
    initial_w3 = weight_vector[4]
    initial_b3 = weight_vector[5]
    initial_w4 = weight_vector[6]
    initial_b4 = weight_vector[7]
else:
    print("No initial weights detected. Initializing to random now.")
    initial_w1, initial_b1, initial_w2, initial_b2, initial_w3, initial_b3, initial_w4, initial_b4 =\
        None, None, None, None, None, None, None, None


def preprocess(obs):
    img = Image.fromarray(obs, 'RGB')
    img = img.convert('L')
    img = img.resize((84,84))
    new_obs = np.array(img)
    return new_obs


def weight_variable(shape, name, initial_weight):
    if initial_weight is None:
        initial = tf.random_normal(shape, stddev=0.01)
        return tf.Variable(initial, name=name)
    else:
        return tf.Variable(initial_weight, name=name)


def bias_variable(shape, name, initial_weight):
    if initial_weight is None:
        initial = tf.constant(0.1, shape=shape, name=name)
        return tf.Variable(initial)
    else:
        return tf.Variable(initial_weight, name=name)


def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='VALID')

# network

sess = tf.Session()
input = tf.placeholder(tf.float32, shape=[None, 84, 84, 4])
conv1_weight = weight_variable(shape=[8, 8, 4, 16], name='conv1_weight', initial_weight=initial_w1)
conv1_bias = bias_variable(shape=[16], name='conv1_bias', initial_weight=initial_b1)
conv1_layer = tf.nn.relu(conv2d(input, conv1_weight, 4) + conv1_bias)
conv2_weight = weight_variable(shape=[4, 4, 16, 32], name='conv2_weight', initial_weight=initial_w2)
conv2_bias = bias_variable(shape=[32], name='conv2_bias', initial_weight=initial_b2)
conv2_layer = tf.nn.relu(conv2d(conv1_layer, conv2_weight, 2) + conv2_bias)
conv2_layer_flattened = tf.reshape(conv2_layer, [-1, 9*9*32])
fc1_weight = weight_variable(shape=[9*9*32, 256], name='fc1_weight', initial_weight=initial_w3)
fc1_bias = bias_variable(shape=[256], name='fc1_bias', initial_weight=initial_b3)
fc1_layer = tf.nn.relu(tf.matmul(conv2_layer_flattened, fc1_weight) + fc1_bias)
fc2_weight = weight_variable(shape=[256, 3], name='fc2_weight', initial_weight=initial_w4)
fc2_bias = bias_variable(shape=[3], name='fc2_bias', initial_weight=initial_b4)
output = tf.matmul(fc1_layer, fc2_weight) + fc2_bias

target = tf.placeholder(tf.float32, None)
action_hot = tf.placeholder('float', [None,3])
action_readout = tf.reduce_sum(tf.mul(output, action_hot), reduction_indices=1)
loss = tf.reduce_mean(tf.square(tf.sub(action_readout, target)))
train_operation = tf.train.RMSPropOptimizer(0.00025, decay=0.95, epsilon=1e-6).minimize(loss)

weights = [conv1_weight, conv1_bias, conv2_weight, conv2_bias, fc1_weight, fc1_bias, fc2_weight, fc2_bias]

### Describe the main training loop ###

# helper function that anneals randomness over first million time steps


def update_randomness(p):
    p = p - .9e-7
    if p < 0.1:
        p = 0.1
    return p


def test_policy(sess):
    env = gym.make('Breakout-v0')
    reward = 0.
    for j in range(100):
        obs1 = env.reset()
        obs2 = env.step(env.action_space.sample())[0]
        obs3 = env.step(env.action_space.sample())[0]
        obs4, _, done, _ = env.step(env.action_space.sample())
        obs1, obs2, obs3, obs4 = preprocess(obs1), preprocess(obs2), preprocess(obs3), preprocess(obs4)
        state = np.transpose([obs1, obs2, obs3, obs4], (1, 2, 0))
        steps = 0.
        while not done and steps < MAX_STEPS:
            Q_val = sess.run(output, feed_dict={input: [state]})
            index = Q_val.argmax()
            if random.random() > 0.03:
                if index == 0:
                    action = 0
                if index == 1:
                    action = 4
                if index == 2:
                    action = 5
            else:
                action = random.choice([4,5])

            new_obs1, reward1, _, _ = env.step(action)
            env.render()
            new_obs2, reward2, _, _ = env.step(action)
            env.render()
            new_obs3, reward3,  _, _ = env.step(action)
            env.render()
            new_obs4, reward4, done, _ = env.step(action)
            env.render()
            obs1 = preprocess(new_obs1)
            obs2 = preprocess(new_obs2)
            obs3 = preprocess(new_obs3)
            obs4 = preprocess(new_obs4)
            new_state = np.transpose([obs1, obs2, obs3, obs4], (1, 2, 0))

            reward += (reward1 + reward2 + reward3 + reward4)*(0.99**steps)
            steps += 1.
        print(reward)
        print(steps)
    avg_reward = reward/100.
    print(avg_reward)
    env.close()
    return avg_reward




env = gym.make('Breakout-v0')
session = tf.Session()
session.run(tf.initialize_all_variables())
target_weights = session.run(weights)
print(target_weights[0])
replay_memory = []
episode_step_count = []
total_steps = 0
prob = 1.0
reward_datapoints = []

for episode in range(1000000):
    obs1 = env.reset()
    obs2 = env.step(env.action_space.sample())[0]
    obs3 = env.step(env.action_space.sample())[0]
    obs4, _, done, _ = env.step(env.action_space.sample())
    obs1, obs2, obs3, obs4 = preprocess(obs1), preprocess(obs2), preprocess(obs3), preprocess(obs4)
    state = np.transpose([obs1, obs2, obs3, obs4], (1, 2, 0))
    steps = 0

    while not done:
        if random.random() > prob:
            Q_vals = session.run(output, feed_dict={input: [state]})
            index = Q_vals.argmax()

        else:
            index = random.choice([0, 1, 2])

        if index == 0:
            action = 0
        if index == 1:
            action = 4
        if index == 2:
            action = 5

        prob = update_randomness(prob)
        obs1 = obs2
        obs2 = obs3
        obs3 = obs4
        new_obs1, reward1, _, _ = env.step(action)
        new_obs2, reward2, _, _ = env.step(action)
        new_obs3, reward3,  _, _ = env.step(action)
        new_obs4, reward4, done, _ = env.step(action)
        obs1 = preprocess(new_obs1)
        obs2 = preprocess(new_obs2)
        obs3 = preprocess(new_obs3)
        obs4 = preprocess(new_obs4)
        new_state = np.transpose([obs1, obs2, obs3, obs4], (1, 2, 0))

        reward = reward1 + reward2 + reward3 + reward4

        if done:
            reward = -1

        replay_memory.append((state, index, reward, new_state, done))
        state = new_state

        if len(replay_memory) > 1000000:
            replay_memory.pop(0)

        if len(replay_memory) >= 50000 and total_steps % 16 == 0:
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
            action_list = np.zeros((32, 3))
            for i in range(32):
                _, action_index, reward, _, terminal = minibatch[i]
                target_q[i] = reward
                if not terminal:
                    target_q[i] = target_q[i]+ 0.99*max_q[i]

                action_list[i][action_index] = 1.0

            states = [m[0] for m in minibatch]
            feed_dict = {input: states, target: target_q, action_hot: action_list}
            session.run(train_operation, feed_dict=feed_dict)



        if total_steps % 10000 == 0:
            target_weights = session.run(weights)
            # avg_reward = test_policy(session)
            # print("average reward: ", str(avg_reward))
            # reward_datapoints.append(avg_reward)
            # np.save('reward_data', reward_datapoints)




        if total_steps % 500000 == 0:
            np.save('weights_' + str(int(total_steps/500000) + 77), target_weights)

        total_steps += 1
        steps += 1

        if done:
            break

    if total_steps > 40000000:
        break



    episode_step_count.append(steps)
    mean_steps = np.mean(episode_step_count[-100:])
    print("Training episode = {}, Total steps = {}, Last 100 mean steps = {}"
          .format(episode, total_steps, mean_steps))
    np.save("episode_step_count", episode_step_count)



target_weights = session.run(weights)
np.save('weights_final', target_weights)
np.save("episode_step_count_final", episode_step_count)


