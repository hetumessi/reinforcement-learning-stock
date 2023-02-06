import json
import os
from random import randint
import pandas as pd
from decimal import Decimal
from flask import Flask, request


# pandas设置最大显示行和列
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 300)

# 调整显示宽度，以便整行显示
pd.set_option('display.width', 1000)

# 显示所有列
pd.set_option('display.max_columns', None)
# 显示所有行
pd.set_option('display.max_rows', None)
# 设置value的显示长度为100，默认为50
pd.set_option('max_colwidth', 100)
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
# from stable_baselines import PPO2, A2C, SAC, ACKTR, DQN, DDPG
from stable_baselines3 import DQN, PPO, A2C, DDPG, HER
# from policies import MlpPolicy
# from vec_env import DummyVecEnv
# from acktr import ACKTR
from config.config import TECHNICAL_INDICATORS_LIST, TRAIN_START_DATE, TEST_START_DATE, TEST_END_DATE, TRAIN_END_DATE
from preprocessors import FeatureEngineer
from rlenv.StockTradingEnv0 import StockTradingEnv

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

initial_money = 10000000
font = fm.FontProperties(fname='font/wqy-microhei.ttc')
# plt.rc('font', family='Source Han Sans CN')
plt.rcParams['axes.unicode_minus'] = False


def data_split(df, start, end):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df.date >= start) & (df.date < end)]
    data = data.sort_values(["date"], ignore_index=True)
    data.index = data.date.factorize()[0]
    return data


def minMaxScaler(allDf):
    return (allDf - allDf.min()) / (allDf.max() - allDf.min())


def discreteScaler(allDf, sections=10):
    obs_indicators = allDf.columns.values.tolist()
    divides = []
    start = 0.0
    for i in range(sections + 1):
        divides.append(Decimal(start).quantize(Decimal("0.1"), rounding="ROUND_HALF_UP"))
        start += 1 / sections
    print(divides)
    print(allDf.head())
    for indicator in obs_indicators:
        allDf[indicator] = pd.cut(allDf[indicator], divides, labels=False) / 10
    return allDf


def dropFields(df, fields):
    for f in fields:
        df.drop(f, axis=1, inplace=True)


# 数据预处理
def prehandle():
    print("==============加载数据===========")
    allDf = pd.read_csv('./stockdata/IF8888.CCFX_day.csv')

    # allDf = pd.read_csv('../FinRL-Library-master/datasets/IF8888.CCFX.csv')
    allDf['date'] = pd.to_datetime(allDf['date'])
    print("==============开始特征工程===========")
    fe = FeatureEngineer(TECHNICAL_INDICATORS_LIST)

    allDf = fe.preprocess_data(allDf)

    print(allDf.head())

    tempdate = allDf.date
    allDf.drop("date", axis=1, inplace=True)

    # rawopen = allDf.open
    rawclose = allDf.close
    # dropFields(allDf, ['open', 'high', 'close','low','volume', 'amount'])
    temp_earnrate = (allDf["close_-1_r"] + 100) / 100
    # allDf = minMaxScaler(allDf) # 即简单实现标准化

    # allDf = discreteScaler(minMaxScaler(allDf))  # 离散化

    allDf.insert(0, 'date', tempdate)
    allDf['earnrate'] = temp_earnrate
    # allDf['rawopen'] = rawopen
    allDf['rawclose'] = rawclose

    return allDf.dropna()


def stock_trade():
    global initial_money
    day_profits = []
    index_profits = []
    actions = []
    allDf = prehandle()

    allDf = pd.read_csv('./discrete_handle.csv')
    # allDf.to_csv('discrete_handle1.csv', index= False)
    print(allDf.head())

    print("==============分割训练集测试集===========")
    df = data_split(allDf, TRAIN_START_DATE, TRAIN_END_DATE)
    df = df.sort_values('date')

    env = StockTradingEnv(df)
    model = A2C("MlpPolicy", env, verbose=1, seed=1)
    model.learn(total_timesteps=int(1e4))
    return
    # model = DQN.load("./IF8888model_DQN")

    model.save('./IF8888model_A2C_20210603')

    df_test = data_split(allDf, TEST_START_DATE, TEST_END_DATE)
    print('总数量', len(df_test))
    # env = DummyVecEnv([lambda: StockTradingEnv(df_test)])
    env = StockTradingEnv(df_test)
    obs = env.reset()
    print('>>>>>>>>>', obs)
    for i in range(len(df_test) - 1):
        # models = [acktr_model, a2c_model]
        # vote = [0, 0, 0]
        # amount_list = [0.0, 0.0, 0.0]
        # actions = []
        # for model in models:
        #     action, _state= model.predict(obs)
        #     actions.append(action)
        #     vote[int(action[0][0])] += 1
        #     amount_list[int(action[0][0])] += action[0][1]
        #
        # print('vote', vote)
        # vote = np.array(vote)
        # major_action = np.argmax(vote)
        #
        # action = [[major_action, amount_list[major_action]/np.max(vote)]]
        #
        # if 2 not in vote:
        #     print('意见不一致')
        #     # action = actions[randint(0, len(actions)-1)]
        #     action = actions[0]
        #
        # print(actions, action)

        action, _state = model.predict(obs)
        actions.append(action)
        obs, rewards, done, info = env.step(action)
        # profit = env.render()
        # if i != 0:
        #     initial_money = initial_money * (df_test.loc[i, "earnrate"])
        # index_profits.append(initial_money - 10000000)
        # day_profits.append(profit)
        if done:
            break
    return day_profits, index_profits, actions


def find_file(path, name):
    # print(path, name)
    for root, dirs, files in os.walk(path):
        for fname in files:
            if name in fname:
                return os.path.join(root, fname)


def test_a_stock_trade(stock_code):
    # stock_file = find_file('./stockdata/train', str(stock_code))

    daily_profits, index_profits, actions = stock_trade()
    # fig, ax = plt.subplots()
    # print('画图', daily_profits)
    # print('基准', index_profits)
    # ax.plot(daily_profits, '-o', label=stock_code, marker='o', ms=10, alpha=0.7, mfc='orange')
    # ax.plot(index_profits, '-o', label="IF8888_INDEX", marker='o', ms=10, alpha=0.7, mfc='green')
    # ax.grid()
    # plt.xlabel('step')
    # plt.ylabel('profit')
    # ax.legend(prop=font)
    # plt.show()
    for action in actions:
        print(action, end=' ')
    # plt.savefig(f'./img/{stock_code}.png')


# @app.route('/getAction')
# def hello():
#     obs = np.array(json.loads(request.args['obs']))
#     return str(int(model.predict(obs)[0]))



if __name__ == '__main__':
    # app.run(host='0.0.0.0',  debug=True)
    test_a_stock_trade('IF8888_DQN.CCFX')
