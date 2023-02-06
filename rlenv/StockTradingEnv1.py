import random
import json
import gym
from gym import spaces
import pandas as pd
import numpy as np

EMPTY = 0
LONG = 1
SHORT = 2


class StockTradingEnv(gym.Env):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, obs, rewards):
        super(StockTradingEnv, self).__init__()

        self.obs = obs
        self.rewards = rewards
        self.reward_range = (-1, 1)

        # 定义动作空间
        # self.action_space = spaces.Discrete(3)
        self.action_space = spaces.Box(
            low=np.array([0]), high=np.array([3]), dtype=np.float16)

        # self.action_space = spaces.Box(
        #     low=np.array([0, 0]), high=np.array([3, 1]), dtype=np.float16)

        # 定义状态空间
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(18,), dtype=np.float16)

    def _next_observation(self):
        obs = self.obs[self.current_step]
        return obs

    def step(self, action):
        action = int(action[0])
        done = False
            # done = True

        # profits
        # reward = self.net_worth - INITIAL_ACCOUNT_BALANCE
        # reward = 1 if reward > 0 else -100

        # 计算奖励
        change = self.rewards[self.current_step]
        print('change', change)
        if (action == LONG and change < 0) or (action == SHORT and change > 0):
            print('negative')
            reward = -abs(change)
        elif (action == LONG and change > 0) or (action == SHORT and change < 0):
            print('positive')
            reward = abs(change)
        else:
            reward = 0
        self.current_step += 1
        if self.current_step >= len(self.rewards) - 1:
            done = True
            # self.current_step = 0  # loop training

        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self, new_df=None):
        # Reset the state of the environment to an initial state

        # pass test dataset to environment
        if new_df:
            self.df = new_df

        # Set the current step to a random point within the data frame
        # self.current_step = random.randint(
        #     0, len(self.df.loc[:, 'open'].values) - 6)
        self.current_step = 1

        return self._next_observation()
