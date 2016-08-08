import tensorflow as tf
import numpy as np
import random
import gym
import matplotlib.pyplot as plt

variance = 1.0

def policy_gradient():
    with tf.variable_scope("policy"):
        params = tf.get_variable("policy_mean_parameters",[4,1])
        state = tf.placeholder("float",[None,4])
        actions = tf.placeholder("float",[None,1])
        advantages = tf.placeholder("float", [None,1])
        mean = tf.matmul(state,params)
        eligibility = -tf.div(tf.square(actions - mean), 2.0*tf.square(variance)) * advantages
        loss = -tf.reduce_sum(eligibility)
        optimizer = tf.train.AdamOptimizer(0.01).minimize(loss)
        return mean, state, actions, advantages, optimizer

def value_gradient():
    with tf.variable_scope("value"):
        state = tf.placeholder("float",[None,4])
        newvals = tf.placeholder("float",[None,1])
        w1 = tf.get_variable("w1",[4,10])
        b1 = tf.get_variable("b1",[10])
        h1 = tf.nn.relu(tf.matmul(state,w1) + b1)
        w2 = tf.get_variable("w2",[10,1])
        b2 = tf.get_variable("b2",[1])
        calculated = tf.matmul(h1,w2) + b2
        diffs = calculated - newvals
        loss = tf.nn.l2_loss(diffs)
        optimizer = tf.train.AdamOptimizer(0.1).minimize(loss)
        return calculated, state, newvals, optimizer, loss

def run_episode(env, policy_grad, value_grad, sess):
    pl_calculated, pl_state, pl_actions, pl_advantages, pl_optimizer = policy_grad
    vl_calculated, vl_state, vl_newvals, vl_optimizer, vl_loss = value_grad
    observation = env.reset()
    totalreward = 0
    states = []
    actions = []
    advantages = []
    transitions = []
    update_vals = []


    for k in range(20000):
        # calculate policy
        obs_vector = np.expand_dims(observation, axis=0)
        action_mean = sess.run(pl_calculated,feed_dict={pl_state: obs_vector})
        action = random.gauss(action_mean[0], variance)
        # record the transition
        states.append(observation)
        actions.append(action)
        # take the action in the environment
        old_observation = observation
        observation, reward, done, info = env.step(action)
        transitions.append((old_observation, action, reward))
        totalreward += reward

        if done:
            break
    for index, trans in enumerate(transitions):
        obs, action, reward = trans

        # calculate discounted monte-carlo return
        future_reward = 0
        future_transitions = len(transitions) - index
        decrease = 1
        for index2 in range(future_transitions):
            future_reward += transitions[(index2) + index][2] * decrease
            decrease = decrease * 0.99
        obs_vector = np.expand_dims(obs, axis=0)
        currentval = sess.run(vl_calculated,feed_dict={vl_state: obs_vector})[0][0]

        # advantage: how much better was this action than normal
        advantages.append(future_reward - currentval)

        # update the value function towards new return
        update_vals.append(future_reward)

    # update value function
    update_vals_vector = np.expand_dims(update_vals, axis=1)
    sess.run(vl_optimizer, feed_dict={vl_state: states, vl_newvals: update_vals_vector})
    # summary = sess.run(summary, feed_dict={vl_state: states, vl_newvals: update_vals_vector})
    # real_vl_loss = sess.run(vl_loss, feed_dict={vl_state: states, vl_newvals: update_vals_vector})

    advantages_vector = np.expand_dims(advantages, axis=1)
    sess.run(pl_optimizer, feed_dict={pl_state: states, pl_advantages: advantages_vector, pl_actions: actions})

    return totalreward, k





env = gym.make('InvertedPendulum-v1')
# env.monitor.start('cartpole-hill/', force=True)
policy_grad = policy_gradient()
value_grad = value_gradient()
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())
#tf.scalar_summary('value loss', value_grad[0])
# tf.scalar_summary('policy loss', policy_grad[5])
# merged_summaries = tf.merge_all_summaries()
# writer = tf.train.SummaryWriter('./value_loss', graph=tf.get_default_graph())

#length, = plt.plot([], [])
lens = []
avgs = [0]



for i in range(2000):
    reward, k = run_episode(env, policy_grad, value_grad, sess)
    # writer.add_summary(tb_summary, global_step=i)
    print('episode {} ran for {} steps and received a total reward of {}'.format(i, k, reward))
    #length.set_xdata(np.append(length.get_xdata(), i))
    #length.set_ydata(np.append(length.get_ydata(), k))
    #plt.draw()
    lens = np.append(lens, k)
    if i > 10:
        last_100_avg = sum(lens[-10:])/10.
        avgs = np.append(avgs, last_100_avg)

        if avgs[-1] > 4000:
            break

    if avgs[-1] > 4000:
        break




    if i % 100 == 0:
        obs = env.reset()
        done = False
        while not done:
            env.render()
            state = np.expand_dims(obs, axis=0)
            action_mean = sess.run(policy_grad[0], feed_dict={policy_grad[1]: state})
            action = random.gauss(action_mean[0][0], variance)
            obs, _, done, _ = env.step(action)
