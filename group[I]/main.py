import numpy as np
import contextlib
import random
from itertools import product
import matplotlib.pyplot as plt
from environment import Environment
import torch
from collections import deque

# Configures numpy print options
@contextlib.contextmanager
def _printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)


class EnvironmentModel:
    def __init__(self, n_states, n_actions, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions

        self.random_state = np.random.RandomState(seed)

    def p(self, next_state, state, action):
        raise NotImplementedError()

    def r(self, next_state, state, action):
        raise NotImplementedError()

    def draw(self, state, action):
        p = [self.p(ns, state, action) for ns in range(self.n_states)]
        next_state = self.random_state.choice(self.n_states, p=p)
        reward = self.r(next_state, state, action)

        return next_state, reward


class Environment(EnvironmentModel):
    def __init__(self, n_states, n_actions, max_steps, pi, seed=None):
        EnvironmentModel.__init__(self, n_states, n_actions, seed)

        self.max_steps = max_steps

        self.pi = pi
        if self.pi is None:
            self.pi = np.full(n_states, 1. / n_states)

    def reset(self):
        self.n_steps = 0
        self.state = self.random_state.choice(self.n_states, p=self.pi)

        return self.state

    def step(self, action):
        if action < 0 or action >= self.n_actions:
            raise Exception('Invalid action.')

        self.n_steps += 1
        done = (self.n_steps >= self.max_steps)

        self.state, reward = self.draw(self.state, action)

        return self.state, reward, done

    def render(self, policy=None, value=None):
        raise NotImplementedError()


class FrozenLake(Environment):
    def __init__(self, lake, slip, max_steps, seed=None):
        """
        lake: A matrix that represents the lake. For example:
        lake =  [['&', '.', '.', '.'],
                ['.', '#', '.', '#'],
                ['.', '.', '.', '#'],
                ['#', '.', '.', '$']]
        slip: The probability that the agent will slip
        max_steps: The maximum number of time steps in an episode
        seed: A seed to control the random number generator (optional)
        """
        # start (&), frozen (.), hole (#), goal ($)
        self.lake = np.array(lake)
        self.lake_flat = self.lake.reshape(-1)

        self.slip = slip

        n_states = self.lake.size + 1
        n_actions = 4

        pi = np.zeros(n_states, dtype=float)
        pi[np.where(self.lake_flat == '&')[0]] = 1.0

        self.absorbing_state = n_states - 1

        # TODO:

        Environment.__init__(self, n_states, n_actions, max_steps, pi, seed=seed)
        # direction
        self.actions = [(-1, 0), (0, -1), (1, 0), (0, 1)]
        self.itos = list(product(range(self.lake.shape[0]), range(self.lake.shape[1])))
        self.stoi = {s: i for (i, s) in enumerate(self.itos)}

        self._p = np.zeros((n_states, n_states, 4))

        for state_index, state in enumerate(self.itos):
            for action_index, action in enumerate(self.actions):
                next_state = (state[0] + action[0], state[1] + action[1])
                if (
                        state_index == 5 or state_index == 7 or state_index == 12 or state_index == 11 or state_index == 15):
                    self._p[state_index, state_index, action_index] = 1.0
                else:
                    next_state_index = self.stoi.get(next_state, state_index)
                    self._p[next_state_index, state_index, action_index] = 1 - self.slip
                    for act in self.actions:
                        next_state_action = (state[0] + act[0], state[1] + act[1])
                        next_state_index = self.stoi.get(next_state_action, state_index)
                        self._p[next_state_index, state_index, action_index] += self.slip / 4

    def step(self, action):
        state, reward, done = Environment.step(self, action)

        done = (state == self.absorbing_state) or done

        return state, reward, done

    def p(self, next_state, state, action):
        return self._p[next_state, state, action]

    def r(self, next_state, state, action):
        if state == 15:
            return 1
        else:
            return 0

    def render(self, policy=None, value=None):
        if policy is None:
            lake = np.array(self.lake_flat)

            if self.state < self.absorbing_state:
                lake[self.state] = '@'

            print(lake.reshape(self.lake.shape))
        else:
            # UTF-8 arrows look nicer, but cannot be used in LaTeX
            # https://www.w3schools.com/charsets/ref_utf_arrows.asp
            actions = ['^', '<', '_', '>']

            print('Lake:')
            print(self.lake)

            print('Policy:')
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.lake.shape))

            print('Value:')
            with _printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))


class Big_frozen_lake(Environment):
    def __init__(self, lake, slip, max_steps, seed=None):
        """
        lake: A matrix that represents the lake. For example:
    big_lake = [['&', '.', '.', '.','.', '.', '.', '.'],
                ['.', '.', '.', '.','.', '.', '.', '.'],
                ['.', '.', '.', '#','.', '.', '.', '.'],
                ['.', '.', '.', '.','.', '#', '.', '.'],
                ['.', '.', '.', '#','.', '.', '.', '.'],
                ['.', '#', '#', '.','.', '.', '#', '.'],
                ['.', '#', '.', '.','#', '.', '#', '.'],
                ['.', '.', '.', '#','.', '.', '.', '$']]
        slip: The probability that the agent will slip
        max_steps: The maximum number of time steps in an episode
        seed: A seed to control the random number generator (optional)
        """
        # start (&), frozen (.), hole (#), goal ($)
        # self.lake = np.zeros(np.array(lake).shape)
        self.lake = np.array(lake)
        self.lake_flat = self.lake.reshape(-1)
        self.slip = slip
        n_states = self.lake.size + 1
        n_actions = 4
        pi = np.zeros(n_states, dtype=float)
        pi[np.where(self.lake_flat == '&')[0]] = 1.0
        self.absorbing_state = n_states - 1

        # TODO:
        Environment.__init__(self, n_states, 4, max_steps, pi, seed)

        # Up, left, down, right.
        self.actions = [(-1, 0), (0, -1), (1, 0), (0, 1)]

        self.itos = list(product(range(self.lake.shape[0]), range(self.lake.shape[1])))
        self.stoi = {s: i for (i, s) in enumerate(self.itos)}

        self._p = np.zeros((n_states, n_states, 4))

        for state_index, state in enumerate(self.itos):
            for action_index, action in enumerate(self.actions):
                next_state = (state[0] + action[0], state[1] + action[1])
                if (
                        state_index == 19 or state_index == 29 or state_index == 35 or state_index == 41 or state_index == 42 or state_index == 46 or state_index == 49 or state_index == 52 or state_index == 54 or state_index == 59):
                    self._p[state_index, state_index, action_index] = 1.0
                else:
                    next_state_index = self.stoi.get(next_state, state_index)
                    self._p[next_state_index, state_index, action_index] = 1 - self.slip
                    for act in self.actions:
                        next_state_action = (state[0] + act[0], state[1] + act[1])
                        next_state_index = self.stoi.get(next_state_action, state_index)
                        self._p[next_state_index, state_index, action_index] += self.slip / 4

    def step(self, action):
        state, reward, done = Environment.step(self, action)
        done = (state == self.absorbing_state) or done
        return state, reward, done

    def p(self, next_state, state, action):
        # TODO:
        return self._p[next_state, state, action]

    def r(self, next_state, state, action):
        # TODO:
        if state == 63:
            return 1
        else:
            return 0
        # return self.lake[self.itos[state]]

    def render(self, policy=None, value=None):
        if policy is None:
            lake = np.array(self.lake_flat)
            if self.state < self.absorbing_state:
                lake[self.state] = '@'

            print(lake.reshape(self.lake.shape))
        else:
            # UTF-8 arrows look nicer, but cannot be used in LaTeX
            # https://www.w3schools.com/charsets/ref_utf_arrows.asp
            actions = ['^', '<', '_', '>']

            print('Lake:')
            print(self.lake)

            print('Policy:')
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.lake.shape))

            print('Value:')
            with _printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))


def play(env):
    actions = ['w', 'a', 's', 'd']

    state = env.reset()
    env.render()

    done = False
    while not done:
        c = input('\nMove: ')
        if c not in actions:
            raise Exception('Invalid action')

        state, r, done = env.step(actions.index(c))

        env.render()
        print('Reward: {0}.'.format(r))


def policy_evaluation(env, policy, gamma, theta, max_iterations):
    value = np.zeros(env.n_states, dtype=float)

    for i in range(max_iterations):
        delta = 0
        for s in range(env.n_states):
            s_value = value[s]
            value[s] = sum(
                [env.p(next_state, s, policy[s]) * (env.r(next_state, s, policy[s]) + gamma * value[next_state]) for next_state in
                 range(env.n_states)])

            delta = max(delta, abs(s_value - value[s]))
        if delta < theta:
            break
    return value


def policy_improvement(env, value, gamma, policy):
    policy_stable = True
    for s in range(env.n_states):
        pol_copy = policy[s].copy()
        v = [
            sum([env.p(next_state, s, a) * (env.r(next_state, s, a) + gamma * value[next_state]) for next_state in range(env.n_states)])
            for a in range(env.n_actions)]
        policy[s] = np.argmax(v)
        if pol_copy != policy[s]:
            policy_stable = False
    return policy_stable


def policy_iteration(env, gamma, theta, max_iterations, policy=None):
    value = np.zeros(env.n_states, dtype=int)
    if policy is None:
        policy = np.zeros(env.n_states, dtype=int)
    else:
        policy = np.array(policy, dtype=int)
    policy_stable = False
    index = 0
    while not policy_stable:
        value = policy_evaluation(env, policy, gamma, theta, max_iterations)
        policy_stable = policy_improvement(env, value, gamma, policy)
        index += 1
    print(index)
    return policy, value


def value_iteration(env, gamma, theta, max_iterations, value=None):
    index = 0
    if value is None:
        value = np.zeros(env.n_states)
    else:
        value = np.array(value, dtype=np.float)
    for _ in range(max_iterations):
        delta = 0.
        for s in range(env.n_states):
            v = value[s]
            value[s] = max([sum(
                [env.p(next_s, s, a) * (env.r(next_s, s, a) + gamma * value[next_s]) for next_s in range(env.n_states)])
                for a in range(env.n_actions)])

            delta = max(delta, np.abs(v - value[s]))

        if delta < theta:
            break

        index = index + 1

    policy = np.zeros(env.n_states, dtype=int)
    for s in range(env.n_states):
        policy[s] = np.argmax([sum(
            [env.p(next_s, s, a) * (env.r(next_s, s, a) + gamma * value[next_s]) for next_s in range(env.n_states)]) for
            a in range(env.n_actions)])

    print(index)
    return policy, value


def sarsa(env, max_episodes, eta, gamma, epsilon, seed=None, optimal_value=None, find_episodes=False):
    random_state = np.random.RandomState(seed)
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    q_values = np.zeros((env.n_states, env.n_actions))
    rewards = []
    # Loop through the maximum number of episodes
    for i in range(max_episodes):
        state = env.reset()
        # Set the terminal flag to False
        terminal = False
        # Select an action using an e-greedy policy based on Q and the current epsilon value
        action = e_greedy(q_values[state], epsilon[i], env.n_actions, random_state)

        # Select action a for state s according to an e-greedy policy based on Q. by random
        while not terminal:
            # Take action and get the next state, reward, and terminal flag
            next_state, reward, terminal = env.step(action)
            # Select the next action using an e-greedy policy based on the Q-values for the next state
            next_action = e_greedy(q_values[next_state], epsilon[i], env.n_actions, random_state)
            # Update the Q-value for state s and action a
            q_values[state][action] = q_values[state][action] + eta[i] * (
                    reward + (gamma * q_values[next_state][next_action]) - q_values[state][action])
            state = next_state
            action = next_action

        # If find_episodes is True, check if the optimal policy has been found
        if find_episodes:
            # Evaluate the current policy using policy evaluation
            value_new = policy_evaluation(env, q_values.argmax(axis=1), gamma, theta=0.001, max_iterations=128)
            # Check if the optimal policy has been found by comparing the values to the optimal value
            if all(abs(optimal_value[i] - value_new[i]) < 0.001 for i in range(len(value_new))):
                print('Episodes to find the optimal policy: {}' + format(i))
                return q_values.argmax(axis=1), value_new

        # Append the sum of max q_values to the rewards array
        rewards.append(sum(q_values.max(axis=1)))

    # Compute the moving average of the rewards using np.convolve
    moving_average = np.convolve(rewards, np.ones(20) / 20, mode='valid')

    # Create a plot showing the episode number on the x-axis and the moving average on the y-axis
    plt.plot(moving_average)
    plt.xlabel('Episode Number')
    plt.ylabel('Moving Average of Returns')
    plt.title('Performance of Sarsa Algorithm')
    plt.show()

    # If the optimal policy was not found, return the policy and value function for the final iteration
    policy = q_values.argmax(axis=1)
    value = q_values.max(axis=1)

    return policy, value


def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None, optimal_value=None, find_episodes=False):
    random_state = np.random.RandomState(seed)
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    q_values = np.zeros((env.n_states, env.n_actions))
    rewards = []

    for i in range(max_episodes):
        # Reset the environment and get the initial state
        state = env.reset()
        # Set the terminal flag to False
        terminal = False
        while not terminal:
            action = e_greedy(q_values[state], epsilon[i], env.n_actions, random_state)
            next_state, reward, terminal = env.step(action)
            next_action = np.argmax(q_values[next_state])
            q_values[state][action] = q_values[state][action] + eta[i] * (
                    reward + (gamma * q_values[next_state][next_action]) - q_values[state][action])
            state = next_state

        if find_episodes:
            value_new = policy_evaluation(env, q_values.argmax(axis=1), gamma, theta=0.001, max_iterations=128)
            if all(abs(optimal_value[i] - value_new[i]) < 0.001 for i in range(len(value_new))):
                print('Episodes to find the optimal policy: ' + format(i))
                return q_values.argmax(axis=1), value_new

        # Append the sum of max q_values to the rewards array
        rewards.append(sum(q_values.max(axis=1)))

    # Compute the moving average of the rewards using np.convolve
    moving_average = np.convolve(rewards, np.ones(20) / 20, mode='valid')

    # Create a plot showing the episode number on the x-axis and the moving average on the y-axis
    plt.plot(moving_average)
    plt.xlabel('Episode Number')
    plt.ylabel('Moving Average of Returns')
    plt.title('Performance of Q-learning Algorithm')
    plt.show()

    policy = q_values.argmax(axis=1)
    value = q_values.max(axis=1)

    return policy, value


class LinearWrapper:
    def __init__(self, env):
        # Save a reference to the environment
        self.env = env

        # Save the number of actions and states in the environment
        self.n_actions = self.env.n_actions
        self.n_states = self.env.n_states
        # Calculate the number of features based on the number of actions and states
        self.n_features = self.n_actions * self.n_states

    def encode_state(self, s):
        # Initialize a matrix of zeros with the same number of rows
        # as actions and the same number of columns as features
        features = np.zeros((self.n_actions, self.n_features))
        for a in range(self.n_actions):
            # Calculate the index for the current state and action
            i = np.ravel_multi_index((s, a), (self.n_states, self.n_actions))
            # Set the corresponding element in the features matrix to 1.0
            features[a, i] = 1.0

        return features

    def decode_policy(self, theta):
        # Initialize the policy and value arrays
        policy = np.zeros(self.env.n_states, dtype=int)
        value = np.zeros(self.env.n_states)

        for s in range(self.n_states):
            # Encode the current state
            features = self.encode_state(s)
            # Calculate the Q-values for the current state using the current theta
            q_values = features.dot(theta)

            # Set the policy for the current state to the action with the highest Q-value
            policy[s] = np.argmax(q_values)
            # Set the value for the current state to the highest Q-value
            value[s] = np.max(q_values)

        return policy, value

    def reset(self):
        return self.encode_state(self.env.reset())

    def step(self, action):
        state, reward, done = self.env.step(action)

        return self.encode_state(state), reward, done

    def render(self, policy=None, value=None):
        self.env.render(policy, value)


def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    theta = np.zeros(env.n_features)
    rewards = []
    for i in range(max_episodes):
        # Reset the environment and get the initial features
        features = env.reset()
        # Calculate the Q-values for the initial features using the current theta
        q_values = features.dot(theta)
        # Select an action using an e-greedy policy based on the Q-values and the current epsilon value
        action = e_greedy(q_values, epsilon[i], env.n_actions, random_state)
        # Set the terminal flag to False
        terminal = False
        while not terminal:
            # Take action and get the next state, reward, and terminal flag
            next_state, reward, terminal = env.step(action)
            # Calculate the temporal difference error
            delta = reward - q_values[action]
            # Calculate the Q-values for the next state using the current theta
            q_values = next_state.dot(theta)
            # Select the next action using an e-greedy policy based on the Q-values for the next state
            action_new = e_greedy(q_values, epsilon[i], env.n_actions, random_state)

            # Update the temporal difference error
            delta = delta + (gamma * max(q_values))
            # Update theta using the temporal difference error
            theta = theta + eta[i] * delta * features[action]
            # Set the current features to the next state
            features = next_state
            # Set the current action to the next action
            action = action_new

        rewards.append(sum(theta))

    moving_average = np.convolve(rewards, np.ones(20) / 20, mode='valid')

    plt.plot(moving_average)
    plt.xlabel('Episode Number')
    plt.ylabel('Moving Average of Returns')
    plt.title('Performance of Linear Sarsa Algorithm')
    plt.show()

    return theta


def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)

    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)
    rewards = []  # Initialize an array to store the rewards for each episode
    for i in range(max_episodes):
        features = env.reset()
        q_values = features.dot(theta)
        terminal = False

        while not terminal:
            action = e_greedy(q_values, epsilon[i], env.n_actions, random_state)
            next_state, reward, terminal = env.step(action)
            delta = reward - q_values[action]
            q_values = next_state.dot(theta)
            delta = delta + (gamma * max(q_values))
            theta = theta + (eta[i] * delta * features[action])
            features = next_state

        # Append the sum of theta to the rewards array
        rewards.append(sum(theta))

    moving_average = np.convolve(rewards, np.ones(20) / 20, mode='valid')

    plt.plot(moving_average)
    plt.xlabel('Episode Number')
    plt.ylabel('Moving Average of Returns')
    plt.title('Performance of Linear Q-learning Algorithm')
    plt.show()
    return theta


def e_greedy(q, epsilon, n_actions, random_state):
    if random.uniform(0, 1) < epsilon:
        a = random_state.choice(np.flatnonzero(q == q.max()))
    else:
        a = random_state.randint(n_actions)
    return a


class FrozenLakeImageWrapper:
    def __init__(self, env):
        self.env = env

        lake = self.env.lake

        self.n_actions = self.env.n_actions
        self.n_states = self.env.lake.size + 1
        self.state_shape = (4, lake.shape[0], lake.shape[1])
        self.absorbing_state = self.n_states - 1

        lake_image = [(lake == c).astype(float) for c in ['&', '#', '$']]

        self.state_image = {self.absorbing_state:
                                np.stack([np.zeros(lake.shape)] + lake_image)}
        for state in range(lake.size):
            x = np.where(lake == state)
            self.state_image[state] = lake_image

    def encode_state(self, state):
        return self.state_image[state]

    def decode_policy(self, dqn):
        states = np.array([self.encode_state(s) for s in range(self.env.n_states)])
        q = dqn(states).detach().numpy()  # torch.no_grad omitted to avoid import

        policy = q.argmax(axis=1)
        value = q.max(axis=1)

        return policy, value

    def reset(self):
        return self.encode_state(self.env.reset())

    def step(self, action):
        state, reward, done = self.env.step(action)

        return self.encode_state(state), reward, done

    def render(self, policy=None, value=None):
        self.env.render(policy, value)


class DeepQNetwork(torch.nn.Module):
    def __init__(self, env, learning_rate, kernel_size, conv_out_channels,
                 fc_out_features, seed):
        torch.nn.Module.__init__(self)
        torch.manual_seed(seed)

        self.conv_layer = torch.nn.Conv2d(in_channels=env.state_shape[0],
                                          out_channels=conv_out_channels,
                                          kernel_size=kernel_size, stride=1)

        h = env.state_shape[1] - kernel_size + 1
        w = env.state_shape[2] - kernel_size + 1

        self.fc_layer = torch.nn.Linear(in_features=h * w * conv_out_channels,
                                        out_features=fc_out_features)
        self.output_layer = torch.nn.Linear(in_features=fc_out_features,
                                            out_features=env.n_actions)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float)

        # Apply the convolutional layer
        x = self.conv_layer(x)

        # Flatten the output of the convolutional layer
        x = x.view(x.size(0), -1)

        # Apply the fully-connected layer
        x = self.fc_layer(x)

        # Use the output layer to generate the final output
        y = self.output_layer(x)

        return y

    def train_step(self, transitions, gamma, tdqn):
        states = np.array([transition[0] for transition in transitions])
        actions = np.array([transition[1] for transition in transitions])
        rewards = np.array([transition[2] for transition in transitions])
        next_states = np.array([transition[3] for transition in transitions])
        dones = np.array([transition[4] for transition in transitions])

        q = self(states)
        q = q.gather(1, torch.Tensor(actions).view(len(transitions), 1).long())
        q = q.view(len(transitions))

        with torch.no_grad():
            next_q = tdqn(next_states).max(dim=1)[0] * (1 - dones)

        target = torch.Tensor(rewards) + gamma * next_q

        # TODO: the loss is the mean squared error between `q` and `target`
        loss = 0

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class ReplayBuffer:
    def __init__(self, buffer_size, random_state):
        self.buffer = deque(maxlen=buffer_size)
        self.random_state = random_state

    def __len__(self):
        return len(self.buffer)

    def append(self, transition):
        self.buffer.append(transition)

    def draw(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return batch


def deep_q_network_learning(env, max_episodes, learning_rate, gamma, epsilon,
                            batch_size, target_update_frequency, buffer_size,
                            kernel_size, conv_out_channels, fc_out_features, seed):
    random_state = np.random.RandomState(seed)
    replay_buffer = ReplayBuffer(buffer_size, random_state)

    dqn = DeepQNetwork(env, learning_rate, kernel_size, conv_out_channels,
                       fc_out_features, seed=seed)
    tdqn = DeepQNetwork(env, learning_rate, kernel_size, conv_out_channels,
                        fc_out_features, seed=seed)

    epsilon = np.linspace(epsilon, 0, max_episodes)

    for i in range(max_episodes):
        state = env.reset()

        done = False
        while not done:
            if random_state.rand() < epsilon[i]:
                action = random_state.choice(env.n_actions)
            else:
                with torch.no_grad():
                    q = dqn(np.array([state]))[0].numpy()

                qmax = max(q)
                best = [a for a in range(env.n_actions) if np.allclose(qmax, q[a])]
                action = random_state.choice(best)

            next_state, reward, done = env.step(action)

            replay_buffer.append((state, action, reward, next_state, done))

            state = next_state

            if len(replay_buffer) >= batch_size:
                transitions = replay_buffer.draw(batch_size)
                dqn.train_step(transitions, gamma, tdqn)

        if (i % target_update_frequency) == 0:
            tdqn.load_state_dict(dqn.state_dict())

    return dqn



def main():
    seed = 0

    # Small lake
    lake = [['&', '.', '.', '.'],
            ['.', '#', '.', '#'],
            ['.', '.', '.', '#'],
            ['#', '.', '.', '$']]

    # Big lake
    big_lake = [['&', '.', '.', '.', '.', '.', '.', '.'],
                ['.', '.', '.', '.', '.', '.', '.', '.'],
                ['.', '.', '.', '#', '.', '.', '.', '.'],
                ['.', '.', '.', '.', '.', '#', '.', '.'],
                ['.', '.', '.', '#', '.', '.', '.', '.'],
                ['.', '#', '#', '.', '.', '.', '#', '.'],
                ['.', '#', '.', '.', '#', '.', '#', '.'],
                ['.', '.', '.', '#', '.', '.', '.', '$']]

    # Small lake
    env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)
    # Big lake
    # env = Big_frozen_lake(big_lake, slip=0.1, max_steps=64, seed=seed)
    print('# Model-based algorithms')
    gamma = 0.9
    theta = 0.001
    max_iterations = 128

    print('')

    print('## Policy iteration')
    policy, value = policy_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)

    opt_value = value.copy()

    print('')

    print('## Value iteration')
    policy, value = value_iteration(env, gamma, theta, max_iterations)
    env.render(policy, value)

    print('')

    print('# Model-free algorithms')
    max_episodes = 10000
    eta = 0.5
    epsilon = 0.5

    print('')

    print('## Sarsa')
    policy, value = sarsa(env, max_episodes, eta, gamma, epsilon, seed=seed, optimal_value=opt_value,
                          find_episodes=False)
    env.render(policy, value)

    print('')

    print('## Q-learning')
    policy, value = q_learning(env, max_episodes, eta, gamma, epsilon, seed=seed, optimal_value=opt_value,
                               find_episodes=False)
    env.render(policy, value)

    linear_env = LinearWrapper(env)

    print('## Linear Sarsa')

    parameters = linear_sarsa(linear_env, max_episodes, eta,
                              gamma, epsilon, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)

    print('')

    print('## Linear Q-learning')

    parameters = linear_q_learning(linear_env, max_episodes, eta,
                                   gamma, epsilon, seed=seed)
    policy, value = linear_env.decode_policy(parameters)
    linear_env.render(policy, value)

    # Part Implemented
    # print('')
    #
    # image_env = FrozenLakeImageWrapper(env)
    #
    # print('## Deep Q-network learning')
    #
    # dqn = deep_q_network_learning(image_env, max_episodes, learning_rate=0.001,
    #                               gamma=gamma, epsilon=0.2, batch_size=32,
    #                               target_update_frequency=4, buffer_size=256,
    #                               kernel_size=3, conv_out_channels=4,
    #                               fc_out_features=8, seed=4)
    # policy, value = image_env.decode_policy(dqn)
    # image_env.render(policy, value)


if __name__ == "__main__":
    main()
