import json
import os
import pickle
from random import randint
import pandas as pd
from stable_baselines3 import DQN, PPO, A2C, DDPG, HER, SAC

from rlenv.StockTradingEnv1 import StockTradingEnv



def stock_trade():
    obs = pickle.load(open("./examples/train_obs.pkl", "rb"))
    rewards = pickle.load(open("./examples/train_reward.pkl", "rb"))
    print(obs, rewards)

    env = StockTradingEnv(obs, rewards)
    model = PPO("MlpPolicy", env, verbose=1, seed=1)
    model.learn(total_timesteps=int(1e4))

    # model = DQN.load("./IF8888model_DQN")

    model.save('./IF8888model_PPO_20210603')





if __name__ == '__main__':
    # app.run(host='0.0.0.0',  debug=True)
    stock_trade()
