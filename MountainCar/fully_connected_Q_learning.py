import gym
import tensorflow as tf
import numpy as np
import random


def policy():
    with tf.variable_scope("policy"):
        input = tf.placeholder("float",[None,2])
        weight1 = tf.get_variable("policy_weight1",[2,6])
        bias1 = tf.get_variable("policy_bias1", [6])
        hidden1 = tf.nn.relu(tf.matmul(input, weight1) + bias1)
        weight2 = tf.get_variable("policy_weight2", [6, 3])
        bias2 = tf.get_variable("policy_bias2", [3])
        actions = tf.placeholder("float", [None, 3])
        adv = tf.placeholder("float", [None, 1])
        output = tf.nn.softmax(tf.matmul(hidden1, weight2) + bias2)
        good_probabilities = tf.reduce_sum(tf.mul(output, actions), reduction_indices=[1])
        eligibility = tf.log(good_probabilities) * adv
        loss = -tf.reduce_sum(eligibility)
        train_optimizer = tf.train.RMSPropOptimizer(0.01).minimize(loss)
        return input, output, actions, adv, train_optimizer


def value():
    with tf.variable_scope("value"):
        input = tf.placeholder("float",[None,2])
        w1 = tf.get_variable("w1",[2,10])
        b1 = tf.get_variable("b1",[10])
        h1 = tf.nn.relu(tf.matmul(input,w1) + b1)
        w2 = tf.get_variable("w2",[10,1])
        b2 = tf.get_variable("b2",[1])
        output = tf.matmul(h1,w2) + b2
        targets = tf.placeholder("float",[None,1])
        diffs = output - targets
        loss = tf.nn.l2_loss(diffs)
        train_optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)
        return input, output, targets, train_optimizer, loss

def update_randomness(expl):
    expl *=.9977
    if expl < 0.1:
        expl = 0.1

    return expl


def run_episode(env, policy, value, sess, expl):
    policy_input, policy_output, policy_actions, policy_adv, policy_optimizer = policy
    value_input, value_output, value_targets, value_optimizer, value_loss = value
    totalreward = 0
    states = []
    actions = []
    adv = []
    transitions = []
    update_vals = []

    obs = env.reset()
    for i in range(1000):
        # calculate policy
        state = np.expand_dims(obs, axis=0)
        prob = sess.run(policy_output,feed_dict={policy_input: state})
        exploration_coin = random.uniform(0,1)
        if exploration_coin < expl:
            action = random.choice([0, 1, 2])

        else:
            policy_coin = random.uniform(0,1)
            if policy_coin < prob[0][0]:
                action = 0
            elif policy_coin > prob[0][0] + prob[0][1]:
                action = 2
            else:
                action = 1

        states.append(obs)
        action_one_hot = np.zeros(3)
        action_one_hot[action] = 1.0
        actions.append(action_one_hot)
        # take the action in the environment
        old_obs = obs
        obs, reward, done, _ = env.step(action)
        if done:
            print("got to the finish line")
            reward = 100
        transitions.append((old_obs, action, reward))
        totalreward += reward


        if done:
            break
    for i1, trans in enumerate(transitions):
        obs, action, reward = trans

        # calculate discounted monte-carlo return
        future_reward = 0
        future_transitions = len(transitions) - i1
        discount = 1
        for i2 in range(future_transitions):
            future_reward += transitions[(i2) + i1][2] * discount
            discount = discount * 0.99
        state = np.expand_dims(obs, axis=0)
        currentval = sess.run(value_output,feed_dict={value_input: state})[0][0]

        adv.append(future_reward - currentval)

        # update the value function towards new return
        update_vals.append(future_reward)

    # update value function
    update_vals_vector = np.expand_dims(update_vals, axis=1)
    sess.run(value_optimizer, feed_dict={value_input: states, value_targets: update_vals_vector})

    adv_vector = np.expand_dims(adv, axis=1)
    sess.run(policy_optimizer, feed_dict={policy_input: states, policy_adv: adv_vector, policy_actions: actions})

    return totalreward


env = gym.make('MountainCar-v0')

# env.monitor.start('mountaincar-hill/', force=True)
policy = policy()
value = value()
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
exploration = 0.5
for i in range(10000):
    reward = run_episode(env, policy, value, sess, exploration)
    exploration = update_randomness(exploration)
    print(i)
    if i % 100 == 0:
        # run an experiment
        env.monitor.start('experiments', force=True)
        obs = env.reset()
        done = False
        for _ in range(1000):
            # env.render()
            state = np.expand_dims(obs, axis=0)
            prob = sess.run(policy[1], feed_dict={policy[0]: state})
            policy_coin = random.uniform(0,1)
            if policy_coin < prob[0][0]:
                action = 0
            elif policy_coin > prob[0][0] + prob[0][1]:
                action = 2
            else:
                action = 1

            obs, _, done, _ = env.step(action)


            if done:
                env.reset()

        env.monitor.close()

done = False

while not done:

    obs = env.reset()
    done = False
    for _ in range(1000):
        env.render()
        state = np.expand_dims(obs, axis=0)
        prob = sess.run(policy[1], feed_dict={policy[0]: state})
        policy_coin = random.uniform(0,1)
        if policy_coin < prob[0][0]:
            action = 0
        elif policy_coin > prob[0][0] + prob[0][1]:
            action = 2
        else:
            action = 1

        obs, reward, done, _ = env.step(action)
        print(reward)







