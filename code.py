import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import itertools


stocks = ['GOOG', 'TSLA', 'AAPL', 'MSFT', 'SPY']

ohlc = {}


def download_data():

    for stock in stocks:

        ticker = yf.Ticker(stock)
        ohlc[stock] = ticker.history(period='2y', interval='1h')['Close']

    return pd.DataFrame(ohlc)


class Env:
    def __init__(self, df):
        self.df = df
        self.n = len(df)
        self.curr_idx = 0
        self.action_space = [0, 1, 2]  # buy, sell, hold
        self.is_invested = 0

        self.state = self.df[feats].to_numpy()
        self.rewards = self.df['SPY'].to_numpy()
        self.total_buy_and_hold = 0

    def reset(self):
        self.curr_idx = 0
        self.total_buy_and_hold = 0
        self.is_invested = 0
        return self.state[self.curr_idx]

    def step(self, action):

        self.curr_idx += 1
        if self.curr_idx >= self.n:
            raise Exception("Episode already done!")

        if action == 0:
            self.is_invested = 1

        elif action == 1:
            self.is_invested = 0

        if self.is_invested:
            reward = self.rewards[self.curr_idx]

        else:
            reward = 0

        next_state = self.state[self.curr_idx]

        self.total_buy_and_hold += self.rewards[self.curr_idx]

        done = (self.curr_idx == self.n - 1)

        return next_state, reward, done


class StateMapper:
    def __init__(self, env, n_bins=6, n_samples=10000):
        states = []
        done = False
        s = env.reset()
        self.D = len(s)
        states.append(s)

        for _ in range(n_samples):
            a = np.random.choice(env.action_space)
            s2, _, done = env.step(a)
            states.append(s2)
            if done:
                s = env.reset()
                states.append(s)

        states = np.array(states)

        self.bins = []
        for d in range(self.D):
            column = np.sort(states[:, d])

            current_bin = []
            for k in range(n_bins):
                boundary = column[int(n_samples/n_bins * (k + 0.5))]
                current_bin.append(boundary)

            self.bins.append(current_bin)

    def transform(self, state):
        x = np.zeros(self.D)
        for d in range(self.D):
            x[d] = int(np.digitize(state[d], self.bins[d]))

        return tuple(x)

    def all_possible_states(self):
        list_of_bins = []
        for d in range(self.D):
            list_of_bins.append(list(range(len(self.bins[d]) + 1)))

        return itertools.product(*list_of_bins)


class Agent:
    def __init__(self, action_size, state_mapper):

        self.action_size = action_size
        self.gamma = 0.8
        self.epsilon = 0.1
        self.learning_rate = 1e-1
        self.state_mapper = state_mapper

        self.Q = {}
        for s in self.state_mapper.all_possible_states():
            s = tuple(s)
            for a in range(self.action_size):
                self.Q[(s, a)] = np.random.randn()

    def act(self, state):

        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.action_size)

        s = self.state_mapper.transform(state)

        act_values = [self.Q[(s, a)] for a in range(self.action_size)]
        return np.argmax(act_values)

    def train(self, state, action, reward, next_state, done):
        s = self.state_mapper.transform(state)
        s2 = self.state_mapper.transform(next_state)

        if done:
            target = reward

        else:
            act_values = [self.Q[(s2, a)] for a in range(self.action_size)]
            target = reward + self.gamma * np.amax(act_values)

        self.Q[(s, action)] += self.learning_rate * \
            (target - self.Q[(s, action)])


def play_one_episode(agent, env, is_train):

    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.act(state)
        next_state, reward, done = env.step(action)
        total_reward += reward
        if is_train:
            agent.train(state, action, reward, next_state, done)

        state = next_state

    return total_reward


if __name__ == '__main__':

    df = download_data()

    df_returns = pd.DataFrame()

    feats = ['GOOG', 'TSLA', 'AAPL', 'MSFT']

    for stock in stocks:
        df_returns[stock] = np.log(df[stock]).diff()

    Ntest = 1000
    train_data = df_returns.iloc[:-Ntest]
    test_data = df_returns.iloc[-Ntest:]

    num_episodes = 500

    train_env = Env(train_data)
    test_env = Env(test_data)

    action_size = len(train_env.action_space)
    state_mapper = StateMapper(train_env)
    agent = Agent(action_size, state_mapper)

    train_rewards = np.empty(num_episodes)
    test_rewards = np.empty(num_episodes)

    for e in range(num_episodes):
        r = play_one_episode(agent, train_env, is_train=True)
        train_rewards[e] = r

        tmp_epsilon = agent.epsilon
        agent.epsilon = 0.
        tr = play_one_episode(agent, test_env, is_train=False)
        agent.epsilon = tmp_epsilon
        test_rewards[e] = tr

        print(f"eps: {e+1}/{num_episodes}, train: {r:.5f}, test: {tr:.5f}")

    plt.plot(train_rewards)
    plt.plot(test_rewards)
    plt.show()
