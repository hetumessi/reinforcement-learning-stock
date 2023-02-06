# from flask import Flask, request
import json
import numpy as np
# from stable_baselines3 import DQN

# app = Flask(__name__)
# model = DQN.load("./IF8888model_DQN")


# @app.route('/getAction')
# def hello():
    # obs = np.array(json.loads(request.args['obs']))
    # res = str(int(model.predict(obs)[0]))
    # print(res)
    # return res

from rqalpha.apis import *

import talib


# 在这个方法中编写任何的初始化逻辑。context对象将会在你的算法策略的任何方法之间做传递
def init(context):
    # context内引入全局变量s1，存储目标合约信息
    context.s1 = 'IF88'

    # 初始化时订阅合约行情。订阅之后的合约行情会在handle_bar中进行更新
    subscribe(context.s1)


# 你选择的期货数据更新将会触发此段逻辑，例如日线或分钟线更新
def handle_bar(context, bar_dict):






def riseup(context):
    # 买
    sell_qty = get_position(context.s1, POSITION_DIRECTION.SHORT).quantity
    # 先判断当前卖方仓位，如果有，则进行平仓操作
    if sell_qty > 0:
        buy_close(context.s1, 1)
    # 买入开仓
    buy_open(context.s1, 1)


def falldown(context):
    # 卖
    buy_qty = get_position(context.s1, POSITION_DIRECTION.LONG).quantity
    # 先判断当前买方仓位，如果有，则进行平仓操作
    if buy_qty > 0:
        sell_close(context.s1, 1)
    # 卖出开仓
    sell_open(context.s1, 1)

# if __name__ == '__main__':
#     app.run(host='0.0.0.0', debug=False)
