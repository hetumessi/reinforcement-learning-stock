from decimal import Decimal

from rqalpha.apis import *
import pandas as pd
import talib
from stable_baselines3 import DQN, PPO, A2C, SAC
from chinese_calendar import is_holiday
from cyutils import prehandle, get_obs
EMPTY = 0
LONG = 1
SHORT = 2

# 在这个方法中编写任何的初始化逻辑。context对象将会在你的算法策略的任何方法之间做传递
def init(context):
    # context内引入全局变量s1，存储目标合约信息
    context.s1 = 'IF88'
    context.model = DQN.load("../IF8888model_DQN_20210514")
    # 初始化时订阅合约行情。订阅之后的合约行情会在handle_bar中进行更新
    subscribe(context.s1)


def transform_date(val):
    str_val = str(val)
    return '-'.join([str_val[:4], str_val[4:6], str_val[6:8]])


# 你选择的期货数据更新将会触发此段逻辑，例如日线或分钟线更新
def handle_bar(context, bar_dict):
    # 大型节假日(不包含周末)提前2天空仓
    after_date = current_snapshot(context.s1).datetime + datetime.timedelta(days=2)
    if is_holiday(after_date) and is_holiday(after_date + datetime.timedelta(days=2)):
        order_to(context.s1, 0)
        return
    price_data = history_bars(context.s1, 21, '1d',
                              ['datetime', 'open', 'high', 'low', 'close', 'volume', 'total_turnover'],
                              include_now=True)
    df = pd.DataFrame(price_data)
    df['datetime'] = df['datetime'].apply(transform_date)
    df['date'] = pd.to_datetime(df['datetime'])
    df.drop("datetime", axis=1, inplace=True)
    df['amount'] = df['total_turnover']
    df.drop("total_turnover", axis=1, inplace=True)
    df = prehandle(df)
    obs = get_obs(df.tail(1))
    res = context.model.predict(obs)[0]
    # res = int(res[0])
    print(res)

    for pos in get_positions():
        if pos.quantity > 0:
            print(pos.quantity, pos.direction)
    # if res == LONG:
    #     close_position(context, 'short')
    #     order(context.s1, 15)
    # if res == SHORT:
    #     close_position(context, 'long')
    #     order(context.s1, -15)
    # if res == EMPTY:
    #     order_to(context.s1, 0)

    if res == 0:
        pass
        # sell_open(context.s1, 2)
    elif res > 0.07:
        close_position(context, 'short')
        order(context.s1, 10)
    elif res > 0.065:
        close_position(context, 'short')
        order(context.s1, 5)
    elif res > 0.06:
        close_position(context, 'short')
        order(context.s1, 3)
    elif res > 0.05:
        close_position(context, 'short')
        order(context.s1, 1)

    elif res < 0:
        order_to(context.s1, 0)
    elif res < 0.01:
        close_position(context, 'long')
        order(context.s1, -10)
    elif res < 0.02:
        close_position(context, 'long')
        order(context.s1, -5)
    elif res < 0.03:
        close_position(context, 'long')
        order(context.s1, -3)

    else:
        order_to(context.s1, 0)


def close_position(context, direction):
    buy_qty = get_position(context.s1, POSITION_DIRECTION.LONG).quantity
    sell_qty = get_position(context.s1, POSITION_DIRECTION.SHORT).quantity
    if buy_qty > 0 and direction == 'long':
        sell_close(context.s1, buy_qty)
    if sell_qty > 0 and direction == 'short':
        buy_close(context.s1, sell_qty)
