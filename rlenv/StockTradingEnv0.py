import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np

INITIAL_ACCOUNT_BALANCE = 10000000

EMPTY = 0
LONG = 1
SHORT = 2


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, df):
        super(StockTradingEnv, self).__init__()

        self.df = df
        self.reward_range = (-1, 1)

        # 定义动作空间
        self.action_space = spaces.Discrete(3)
        # self.action_space = spaces.Box(
        #     low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)


        # 定义状态空间
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(18,), dtype=np.float16)

    def _next_observation(self):
        obs_indicators = self.df.columns.values.tolist()
        obs_indicators.remove('date')
        obs_indicators.remove('earnrate')
        # obs_indicators.remove('rawopen')
        obs_indicators.remove('rawclose')

        obs_indicators.remove('open')
        obs_indicators.remove('high')
        obs_indicators.remove('low')
        obs_indicators.remove('volume')
        obs_indicators.remove('amount')


        obs = np.array([
            self.df.loc[self.current_step, x] for x in obs_indicators
        ])
        return obs

    def _take_action(self, action):
        print(action)
        # Set the current price to a random price within the time step
        # current_price = random.uniform(
        #     self.df.loc[self.current_step, "rawopen"], self.df.loc[self.current_step, "rawclose"])
        current_price = self.df.loc[self.current_step, "rawclose"]
        # action_type = int(action)
        # if action > 0.05:
        # if action == LONG:
        #     print('买')
        #     # 用当前现金除以每股价格计算可以买多少
        #     shares_bought = int(self.balance / current_price)
        #     # 计算花费
        #     additional_cost = shares_bought * current_price
        #     # 扣减现金
        #     self.balance -= additional_cost
        #     # 更新持股数量
        #     self.shares_held += shares_bought
        #
        # # elif action_type == 1:
        # else:
        #     print('卖')
        #     shares_sold = self.shares_held
        #     self.balance += shares_sold * current_price
        #     self.shares_held -= shares_sold

        self.current_step += 1

        current_price = self.df.loc[self.current_step, "rawclose"]

        # current_price = random.uniform(
        #     self.df.loc[self.current_step, "rawopen"], self.df.loc[self.current_step, "rawclose"])
        self.net_worth = self.balance + self.shares_held * current_price

        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth


    def step(self, action):
        print('执行')
        # 上个时间步价格
        index_before = self.df.loc[self.current_step, 'rawclose']

        self._take_action(action)
        done = False

        # self.current_step += 1
        print('当前', self.current_step, len(self.df))
        if self.current_step >= len(self.df) - 1:
            self.current_step = 0  # loop training
            # done = True

        # profits
        # reward = self.net_worth - INITIAL_ACCOUNT_BALANCE
        # reward = 1 if reward > 0 else -100

        index_after = self.df.loc[self.current_step, 'rawclose']
        print('之后基准', index_after)

        # 计算奖励
        change = (index_after - index_before) / index_before

        if (action == LONG and change < 0) or (action == SHORT and change > 0):
            reward = -1
        elif (action == LONG and change > 0) or (action == SHORT and change < 0):
            reward = 1
        else:
            reward = 0

        if self.net_worth <= 0:
            done = True

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self, new_df=None):
        # Reset the state of the environment to an initial state
        self.balance = INITIAL_ACCOUNT_BALANCE
        self.net_worth = INITIAL_ACCOUNT_BALANCE
        self.max_net_worth = INITIAL_ACCOUNT_BALANCE
        self.shares_held = 0

        # pass test dataset to environment
        if new_df:
            self.df = new_df

        # Set the current step to a random point within the data frame
        # self.current_step = random.randint(
        #     0, len(self.df.loc[:, 'open'].values) - 6)
        self.current_step = 0

        return self._next_observation()

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        print('净值', self.net_worth)
        profit = self.net_worth - INITIAL_ACCOUNT_BALANCE
        print('-' * 30)
        print(f'Step: {self.current_step}')
        print(f'Balance: {self.balance}')
        print(f'Shares held: {self.shares_held}')
        print(f'Net worth: {self.net_worth} (Max net worth: {self.max_net_worth})')
        print(f'Profit: {profit}')
        return profit * 4
