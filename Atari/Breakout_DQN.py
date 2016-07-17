import gym
import tensorflow as tf
import numpy as np
import random
from PIL import Image

## The following is a 1 layer fully connected + softmax
## neural net. As input it takes the 4 observation values of
## the cartpole simulator and outputs 0 or 1 at each time
## step, which controls left and right movement of the cart.


def make_network():
    # Initializes tensorflow graph. Input is 4 84x84 B&W array shaped as [84,84,4]
    input = tf.placeholder('float', [None, 84, 84, 4])
    layer1_weights = tf.Variable(tf.truncated_normal([8,8,4,16], stddev=0.01))
    layer1_bias = tf.Variable(tf.constant(0.1, shape=[16]))
    conv1 = tf.nn.relu(tf.nn.conv2d(input, layer1_weights, strides=[1,4,4,1], padding='VALID') + layer1_bias)
    layer2_weights = tf.Variable(tf.truncated_normal([4,4,16,32], stddev=0.01))
    layer2_bias = tf.Variable(tf.constant(0.1, shape=[32]))
    conv2 = tf.nn.relu(tf.nn.conv2d(conv1, layer2_weights, strides=[1,2,2,1], padding='VALID') + layer2_bias)
    conv2_reshape = tf.reshape(conv2, [-1, 2592])
    layer3_weights = tf.Variable(tf.truncated_normal([2592, 256], stddev=0.01))
    layer3_bias = tf.Variable(tf.constant(0.1, shape=[256]))
    layer3 = tf.nn.relu(tf.matmul(conv2_reshape, layer3_weights)+layer3_bias)
    layer4_weights = tf.Variable(tf.truncated_normal([256, 6], stddev=0.01))
    layer4_bias = tf.Variable(tf.constant(0.1, shape=[6]))
    output = tf.matmul(layer3, layer4_weights) + layer4_bias

    weights = [layer1_weights, layer2_weights, layer3_weights, layer4_weights,
               layer1_bias, layer2_bias, layer3_bias, layer4_bias]

    targets = tf.placeholder('float', None)
    action = tf.placeholder('float', [None,6])
    action_readout = tf.reduce_sum(tf.mul(output, action), reduction_indices=1)
    loss = tf.reduce_mean(tf.square(tf.sub(action_readout, targets)))
    train_operation = tf.train.AdamOptimizer(0.01).minimize(loss)

    return input, action, weights, targets, output, train_operation


def preprocess(obs):
    img = Image.fromarray(obs, 'RGB')
    img = img.convert('L')
    img = img.resize((84,84))
    new_obs = np.array(img)
    return new_obs

def train(input, action_one_hots, weights, targets, output, train_operation, num_episodes=1000, max_steps=800):
    # Contains full training procedure. Main loop over num of episodes.
    # state will consist of previous 4 frames.
    # actions determined by eps. greedy policy w.r.t current Q-values
    # state action reward tuples will be collected and placed in a replay memory.
    # minibatch samples will be drawn from replay memory for training
    env = gym.make('Breakout-v0')
    session = tf.Session()
    session.run(tf.initialize_all_variables())
    target_weights = session.run(weights)
    replay_memory = []
    episode_step_count = []
    episode_count = 0
    total_steps = 0

    for episode in range(num_episodes):
        obs1 = env.reset()
        obs2, _, _, _ = env.step(env.action_space.sample())
        obs3, _, _, _ = env.step(env.action_space.sample())
        obs4, _, _, _ = env.step(env.action_space.sample())
        obs1, obs2, obs3, obs4 = preprocess(obs1), preprocess(obs2), preprocess(obs3), preprocess(obs4)
        state = [obs1, obs2, obs3, obs4]
        state = np.transpose(state, (1, 2, 0))
        print(np.shape(state))
        steps = 0

        for step in range(max_steps):
            if random.random() > 0.1:
                Q_vals = session.run(output, feed_dict={input: [state]})
                action = Q_vals.argmax()
            else:
                action = env.action_space.sample()

            obs1 = obs2
            obs2 = obs3
            obs3 = obs4
            obs4, reward, done, info = env.step(action)
            obs4 = preprocess(obs4)
            new_state = [obs1, obs2, obs3, obs4]
            new_state = np.transpose(new_state, (1, 2, 0))

            if done:
                reward = -100

            replay_memory.append((state, action, reward, new_state, done))
            state = new_state

            if len(replay_memory) > 10000:
                replay_memory.pop(0)

            if len(replay_memory) >= 50:
                # here is where the training procedure takes place

                # compute the one step q-values w.r.t. old weights (ie y in the loss function (y-Q(s,a,0))^2)
                # Also defines the one-hot readout action vectors
                minibatch = random.sample(replay_memory, 50)
                next_states = [m[3] for m in minibatch]
                feed_dict = {input: next_states}
                feed_dict.update(zip(weights, target_weights))
                q_vals = session.run(output, feed_dict=feed_dict)
                max_q = q_vals.max(axis=1)
                target_q = np.zeros(50)
                action_list = np.zeros((50,6))
                for i in range(50):
                    _, action_index, reward, _, terminal = minibatch[i]
                    target_q[i] = reward
                    if not terminal:
                        target_q[i] =+ 0.99*max_q[i]

                    action_list[i][action_index] = 1.0

                states = [m[0] for m in minibatch]
                feed_dict = {input: states, targets: target_q, action_one_hots: action_list}
                session.run(train_operation, feed_dict=feed_dict)

            total_steps += 1
            steps += 1

            if done:
                break

        episode_step_count.append(steps)
        mean_steps = np.mean(episode_step_count[-100:])
        print("Training episode = {}, Total steps = {}, Last 100 mean steps = {}"
              .format(episode, total_steps, mean_steps))

        if episode % 100 == 0:
            target_weights = session.run(weights)



def test_network():
    # create a new env
    # play using greedy policy w.r.t. Q values
    # record episode length
    return True

def update_randomeness():
    # can add this in if you want to anneal randomness. Without it we'll keep constant randomness .1
    return True


# i, a, w, t, o, tr = make_network()
# train(i, a, w, t, o, tr)
