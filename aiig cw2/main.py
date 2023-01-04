import numpy as np
import contextlib
import random
from itertools import product

from task1 import Environment

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

        #TODO:

        Environment.__init__(self, n_states, n_actions, max_steps, pi, seed=seed)
        #direction
        self.actions=[(-1,0),(0,-1),(1,0),(0,1)]
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
        return self._p[next_state,state,action]


    #TODO:

    def r(self, next_state, state, action):
        if state==15:
            return 1
        else:
            return 0

    #TODO:

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
    value = np.zeros(env.n_states, dtype=np.float)

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

    for i in range(max_episodes):
        s = env.reset()
        terminal = False
        a = e_greedy(q[s], epsilon[i], env.n_actions, random_state)

        # Select action a for state s according to an e-greedy policy based on Q. by random
        while not terminal:
            next_s, r, terminal = env.step(a)
            next_a = e_greedy(q[next_s], epsilon[i], env.n_actions, random_state)
            q[s][a] = q[s][a] + eta[i] * (r + (gamma * q[next_s][next_a]) - q[s][a])
            s = next_s
            a = next_a

        if find_episodes:
            value_new = policy_evaluation(env, q.argmax(axis=1), gamma, theta=0.001, max_iterations=100)
            if all(abs(optimal_value[i] - value_new[i]) < 0.001 for i in range(len(value_new))):
                print('Episodes to find the optimal policy: ' + str(i))
                return q.argmax(axis=1), value_new

    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value

def e_greedy(q, epsilon, n_actions, random_state):
    if random.uniform(0, 1) < epsilon:
        a = random_state.choice(np.flatnonzero(q == q.max()))
    else:
        a = random_state.randint(n_actions)
    return a



def q_learning(env, max_episodes, eta, gamma, epsilon, seed=None, optimal_value=None, find_episodes=False):
    random_state = np.random.RandomState(seed)
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)
    q = np.zeros((env.n_states, env.n_actions))
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
            value_new = policy_evaluation(env, q.argmax(axis=1), gamma, theta=0.001, max_iterations=100)
            if all(abs(optimal_value[i]-value_new[i]) < 0.001 for i in range(len(value_new))):
                print('Episodes to find the optimal policy: ' + str(i))
                return q.argmax(axis=1), value_new

    policy = q.argmax(axis=1)
    value = q.max(axis=1)

    return policy, value


def e_greedy(q, epsilon, n_actions, random_state):
    if random.uniform(0, 1) < epsilon:
        a = random_state.choice(np.flatnonzero(q == q.max()))
    else:
        a = random_state.randint(n_actions)
    return a

def main():
    seed=0;
    # Small lake
    lake = [['&', '.', '.', '.'],
            ['.', '#', '.', '#'],
            ['.', '.', '.', '#'],
            ['#', '.', '.', '$']]
    env=FrozenLake(lake,slip=0.1,max_steps=16,seed=seed)
    play(env)
    print('# Model-based algorithms')
    gamma = 0.9
    theta = 0.001
    max_iterations = 100

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
    max_episodes = 40000
    eta = 0.1
    epsilon = 0.9

    print('')

    print('## Sarsa')
    policy, value = sarsa(env, max_episodes, eta, gamma, epsilon, seed=seed, optimal_value=opt_value,
                          find_episodes=False)
    env.render(policy, value)

    print('')
    print(opt_value)
    print(policy_evaluation(env, policy, gamma, theta=0.001, max_iterations=100))

    print('## Q-learning')
    policy, value = q_learning(env, max_episodes, eta, gamma, epsilon, seed=seed, optimal_value=opt_value,
                               find_episodes=False)
    env.render(policy, value)


if __name__ == "__main__":
    main()


