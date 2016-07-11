import gym
import tensorflow as tf
import numpy as np
import random

## The following is a 1 layer fully connected + softmax
## neural net. As input it takes the 4 observation values of
## the cartpole simulator and outputs 0 or 1 at each time
## step, which controls left and right movement of the cart.

class MC_Controller:
    FUTURE_DISCOUNT = 0.99
    RANDOM_ACTION_DECAY = 0.99
    MIN_RANDOM_ACTION_PROBABILITY = 0.1
    HIDDEN1_SIZE = 4
    HIDDEN2_SIZE = 4
    NUM_EPISODES = 200
    MAX_STEPS = 300
    LEARNING_RATE = 0.001
    MINIBATCH_SIZE = 50
    REG_FACTOR = 0.001
    RANDOM_ACTION_PROBABILITY = 0.5
    WEIGHT_UPDATE_FREQUENCY = 10
    MEMORY = []
    MEMORY_CAPACITY = 10000
    LOG_DIR = '/tmp/MCController'

    def __init__(self):
        self.env = gym.make('MountainCar-v0')

    def make_network(self):
        # define input, output, weight, and bias variables for network
        self.input = tf.placeholder("float", [None, 2])
        self.layer_1_weights = tf.Variable(tf.truncated_normal([2, self.HIDDEN1_SIZE],stddev=0.01))
        self.layer_1_biases = tf.Variable(tf.constant(0.01, shape=[self.HIDDEN1_SIZE]))
        self.layer_2_weights = tf.Variable(tf.truncated_normal([self.HIDDEN1_SIZE, self.HIDDEN2_SIZE],stddev=0.01))
        self.layer_2_biases = tf.Variable(tf.constant(0.01, shape=[self.HIDDEN2_SIZE]))
        self.output_layer_weights = tf.Variable(tf.truncated_normal([self.HIDDEN2_SIZE, 3],stddev=0.01))
        self.output_layer_biases = tf.Variable(tf.constant(0.01, shape=[3]))


        # define layers
        self.hidden_layer_1 = tf.nn.tanh(tf.add(tf.matmul(self.input, self.layer_1_weights), self.layer_1_biases))
        self.hidden_layer_2 = tf.nn.tanh(tf.add(tf.matmul(self.hidden_layer_1, self.layer_2_weights), self.layer_2_biases))
        self.output = tf.add(tf.matmul(self.hidden_layer_2, self.output_layer_weights), self.output_layer_biases)
        self.action = tf.placeholder("float", [None,3])
        self.targets = tf.placeholder("float", [None])

        # weights and readout will be useful for calculations
        self.weights = [self.layer_1_weights, self.layer_1_biases,
                        self.layer_2_weights, self.layer_2_biases,
                        self.output_layer_weights, self.output_layer_biases]
        self.readout_action = tf.reduce_sum(tf.mul(self.output, self.action), reduction_indices=1)

        # define loss and training method
        self.loss = tf.reduce_mean(tf.square(tf.sub(self.readout_action,self.targets)))
        for w in [self.layer_1_weights, self.layer_2_weights]:
            self.loss+=self.REG_FACTOR*tf.reduce_sum(tf.square(w))
        self.train_operation = tf.train.AdamOptimizer(self.LEARNING_RATE).minimize(self.loss)

    # train will run a full episode of CartPole and construct a batch from it
    # upon completion of the episode (or perhaps some # time steps at most), the
    # network will train on the data
    def train(self, num_episodes=NUM_EPISODES):
        self.session=tf.Session()

        # set up Tensorboard summary
        tf.scalar_summary('loss', self.loss)
        self.summary = tf.merge_all_summaries()
        self.summary_writer = tf.train.SummaryWriter(self.LOG_DIR, self.session.graph)

        self.session.run(tf.initialize_all_variables())

        total_steps = 0
        episode_step_counts = []
        target_weights = self.session.run(self.weights)

        for episode in range(num_episodes):
            state = self.env.reset()
            steps = 0
            for step in range(self.MAX_STEPS):
                if random.random() > self.RANDOM_ACTION_PROBABILITY:
                    q_values = self.session.run(self.output, feed_dict={self.input: [state]})
                    action =q_values.argmax()
                else:
                    action = self.env.action_space.sample()

                self.update_policy_randomness()
                obs, reward, done, info = self.env.step(action)

                if done:
                    reward = 100.0

                self.MEMORY.append((state, action, reward, obs, done))

                if(len(self.MEMORY) > self.MEMORY_CAPACITY):
                    self.MEMORY.pop(0)

                state = obs

                if(len(self.MEMORY) >= 50):
                    minibatch = random.sample(self.MEMORY, 50)
                    next_states = [m[3] for m in minibatch]
                    feed_dict = {self.input: next_states}
                    feed_dict.update(zip(self.weights, target_weights))
                    q_values = self.session.run(self.output, feed_dict=feed_dict)
                    max_q_values = q_values.max(axis=1)

                    # compute target q values
                    target_q_values = np.zeros(self.MINIBATCH_SIZE)
                    action_list = np.zeros((self.MINIBATCH_SIZE,3))
                    for i in range(self.MINIBATCH_SIZE):
                        _, action, reward, _, terminal = minibatch[i]
                        target_q_values[i] = reward
                        if not terminal:
                            target_q_values[i] += self.FUTURE_DISCOUNT*max_q_values[i]
                        action_list[i][action] = 1.0

                    states = [m[0] for m in minibatch]
                    self.session.run(self.train_operation, feed_dict={
                        self.input: states,
                        self.targets: target_q_values,
                        self.action: action_list
                    })

                total_steps += 1
                steps += 1

                if done:
                    break

            episode_step_counts.append(steps)
            mean_steps = np.mean(episode_step_counts[-100:])
            print("Training episode = {}, Total steps = {}, Last-100 mean steps = {}"
                  .format(episode, total_steps, mean_steps))

            if episode % self.WEIGHT_UPDATE_FREQUENCY == 0:
                target_weights = self.session.run(self.weights)

    def update_policy_randomness(self):
        self.RANDOM_ACTION_PROBABILITY *= self.RANDOM_ACTION_DECAY
        if self.RANDOM_ACTION_PROBABILITY < self.MIN_RANDOM_ACTION_PROBABILITY:
            self.RANDOM_ACTION_PROBABILITY = self.MIN_RANDOM_ACTION_PROBABILITY

    def control(self):
        state = self.env.reset()
        done = False
        steps = 0
        while not done and steps < 200:
            self.env.render()
            q_values = self.session.run(self.output, feed_dict={self.input: [state]})
            action = q_values.argmax()
            state, _, done, _ = self.env.step(action)
            steps += 1
        return steps


controller = MC_Controller()
controller.make_network()
controller.train()

for i in range(100):
    steps = controller.control()
    print(steps)


