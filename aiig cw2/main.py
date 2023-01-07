import numpy as np
import contextlib
import random
from itertools import product
import matplotlib.pyplot as plt
from environment import Environment


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
                if (state_index == 19 or state_index == 29 or state_index == 35 or state_index == 41 or state_index == 42 or state_index == 46 or state_index == 49 or state_index == 52 or state_index == 54 or state_index == 59):
                    self._p[state_index, state_index, action_index] = 1.0
                else:
                    next_state_index = self.stoi.get(next_state, state_index)
                    self._p[next_state_index, state_index, action_index] = 1 - self.slip
                    for act in self.actions:
                        next_state_action = (state[0] + act[0], state[1] + act[1])
                        next_state_index = self.stoi.get(next_state_action, state_index)
                        self._p[next_state_index, state_index, action_index] += self.slip/4

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

    for _ in range(max_iterations):
        delta = 0
        for s in range(env.n_states):
            v = value[s]
            value[s] = sum(
                [env.p(next_s, s, policy[s]) * (env.r(next_s, s, policy[s]) + gamma * value[next_s]) for next_s in
                 range(env.n_states)])

            delta = max(delta, abs(v - value[s]))
        if delta < theta:
            break
    return value


def policy_improvement(env, value, gamma, policy):
    policy_stable = True
    for s in range(env.n_states):
        pol = policy[s].copy()
        v = [
            sum([env.p(next_s, s, a) * (env.r(next_s, s, a) + gamma * value[next_s]) for next_s in range(env.n_states)])
            for a in range(env.n_actions)]
        policy[s] = np.argmax(v)
        if pol != policy[s]:
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
    q = np.zeros((env.n_states, env.n_actions))
    rewards = []
    # Loop through the maximum number of episodes
    for i in range(max_episodes):
        s = env.reset()
        # Set the terminal flag to False
        terminal = False
        # Select an action using an e-greedy policy based on Q and the current epsilon value
        a = e_greedy(q[s], epsilon[i], env.n_actions, random_state)

        # Select action a for state s according to an e-greedy policy based on Q. by random
        while not terminal:
            # Take action a and get the next state, reward, and terminal flag
            next_s, r, terminal = env.step(a)
            # Select the next action using an e-greedy policy based on the Q-values for the next state
            next_a = e_greedy(q[next_s], epsilon[i], env.n_actions, random_state)
            # Update the Q-value for state s and action a
            q[s][a] = q[s][a] + eta[i] * (r + (gamma * q[next_s][next_a]) - q[s][a])
            s = next_s
            a = next_a

        # If find_episodes is True, check if the optimal policy has been found
        if find_episodes:
            print('yeah')
            # Evaluate the current policy using policy evaluation
            value_new = policy_evaluation(env, q.argmax(axis=1), gamma, theta=0.001, max_iterations=128)
            # Check if the optimal policy has been found by comparing the values to the optimal value
            if all(abs(optimal_value[i] - value_new[i]) < 0.001 for i in range(len(value_new))):
                print('Episodes to find the optimal policy: {}' + format(i))
                return q.argmax(axis=1), value_new

        rewards.append(sum(q.max(axis=1)))
    # If the optimal policy was not found, return the policy and value function for the final iteration
    policy = q.argmax(axis=1)
    value = q.max(axis=1)
    moving_average = np.convolve(rewards, np.ones(20) / 20, mode='valid')
    plt.plot(moving_average)
    plt.xlabel('Episode Number')
    plt.ylabel('Moving Average of Returns')
    plt.title('Performance of Sarsa Algorithm')
    plt.show()
    return policy, value


def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None, optimal_value=None, find_episodes=False):
    random_state = np.random.RandomState(seed)
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    q = np.zeros((env.n_states, env.n_actions))
    rewards = []
    for i in range(max_episodes):
        s = env.reset()
        terminal = False
        while not terminal:
            a = e_greedy(q[s], epsilon[i], env.n_actions, random_state)
            next_s, r, terminal = env.step(a)
            next_a = np.argmax(q[next_s])
            q[s][a] = q[s][a] + eta[i] * (r + (gamma * q[next_s][next_a]) - q[s][a])
            s = next_s

        if find_episodes:
            value_new = policy_evaluation(env, q.argmax(axis=1), gamma, theta=0.001, max_iterations=128)
            if all(abs(optimal_value[i] - value_new[i]) < 0.001 for i in range(len(value_new))):
                print('Episodes to find the optimal policy: ' + format(i))
                return q.argmax(axis=1), value_new

        rewards.append(sum(q.max(axis=1)))
    policy = q.argmax(axis=1)
    value = q.max(axis=1)
    moving_average = np.convolve(rewards, np.ones(20) / 20, mode='valid')
    plt.plot(moving_average)
    plt.xlabel('Episode Number')
    plt.ylabel('Moving Average of Returns')
    plt.title('Performance of Q-learning Algorithm')
    plt.show()
    return policy, value


################ Non-tabular model-free algorithms ################

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
            q = features.dot(theta)

            # Set the policy for the current state to the action with the highest Q-value
            policy[s] = np.argmax(q)
            # Set the value for the current state to the highest Q-value
            value[s] = np.max(q)

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
    rewards=[]
    for i in range(max_episodes):

        features = env.reset()
        q = features.dot(theta)
        # Select an action using an e-greedy policy based on the Q-values and the current epsilon value
        a = e_greedy(q, epsilon[i], env.n_actions, random_state)
        # Set the terminal flag to False
        terminal = False
        while not terminal:
            # Take action a and get the next state, reward, and terminal flag
            next_s, r, terminal = env.step(a)
            # Calculate the temporal difference error
            delta = r - q[a]
            # Calculate the Q-values for the next state using the current theta
            q = next_s.dot(theta)
            # Select the next action using an e-greedy policy based on the Q-values for the next state
            a_new = e_greedy(q, epsilon[i], env.n_actions, random_state)

            # Update the temporal difference error
            delta = delta + (gamma * max(q))
            # Update theta using the temporal difference error
            theta = theta + eta[i] * delta * features[a]
            # Set the current features to the next state
            features = next_s
            # Set the current action to the next action
            a = a_new

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
    rewards=[]
    for i in range(max_episodes):
        features = env.reset()
        q = features.dot(theta)
        terminal = False

        while not terminal:
            a = e_greedy(q, epsilon[i], env.n_actions, random_state)
            next_s, r, terminal = env.step(a)
            delta = r - q[a]
            q = next_s.dot(theta)
            delta = delta + (gamma * max(q))
            theta = theta + (eta[i] * delta * features[a])
            features = next_s
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


def main():
    seed = 0;

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

    # env = FrozenLake(lake, slip=0.1, max_steps=16, seed=seed)
    # env = Big_frozen_lake(big_lake, slip=0.1, max_steps=64, seed=seed)
    # play(env)
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


if __name__ == "__main__":
    main()
