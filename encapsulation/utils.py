from decimal import Decimal
import numpy as np
import pandas as pd
# 将DataFrame转为强化学习输入层需要的格式
from stockstats import StockDataFrame as Sdf
TECHNICAL_INDICATORS_LIST = ["macdh",
                             "boll",
                             "boll_ub",
                             "boll_lb",
                             "close_5_sma",
                             "close_10_sma",
                             "close_20_sma",
                             "rsi_6",
                             "rsi_12",
                             "cci",
                             "wr_10",
                             "open_-1_r",
                             "high_-1_r",
                             "low_-1_r",
                             "amount_-1_r",
                             "volume_-1_r",
                             "close_-1_r"]
class FeatureEngineer:

    def __init__(
        self,
        tech_indicator_list
    ):
        self.tech_indicator_list = tech_indicator_list

    def preprocess_data(self, df):
        """main method to do the feature engineering
        @:param config: source dataframe
        @:return: a DataMatrices object
        """
        # add technical indicators using stockstats
        df = self.add_technical_indicator(df)
        print("Successfully added technical indicators")

        # fill the missing values at the beginning and the end
        df = df.fillna(method="bfill").fillna(method="ffill")
        return df

    def add_technical_indicator(self, data):
        """
        calculate technical indicators
        use stockstats package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df = df.sort_values(by=['date'])
        stock = Sdf.retype(df.copy())

        for indicator in self.tech_indicator_list:
            indicator_df = pd.DataFrame()
            try:
                temp_indicator = stock[indicator]
                temp_indicator = pd.DataFrame(temp_indicator)
                temp_indicator['date'] = df['date'].to_list()
                indicator_df = indicator_df.append(
                    temp_indicator, ignore_index=True
                )
            except Exception as e:
                print(e)
            df = df.merge(indicator_df[['date',indicator]],on=['date'],how='left')
        df = df.sort_values(by=['date'])
        return df




def get_obs(df):
    obs_indicators = df.columns.values.tolist()
    obs_indicators.remove('date')
    obs_indicators.remove('open')
    obs_indicators.remove('high')
    obs_indicators.remove('low')
    obs_indicators.remove('volume')
    obs_indicators.remove('amount')

    obs = np.array([
        df.iloc[0][x] for x in obs_indicators
    ])
    return obs

# 最小最大值归一化
def minMaxScaler(allDf):
    return (allDf - allDf.min()) / (allDf.max() - allDf.min())

# 分段离散化处理
def discreteScaler(allDf, sections=10):
    obs_indicators = allDf.columns.values.tolist()
    divides = []
    start = 0.0
    for i in range(sections + 1):
        divides.append(Decimal(start).quantize(Decimal("0.1"), rounding="ROUND_HALF_UP"))
        start += 1 / sections
    # print(divides)
    # print(allDf.head())
    for indicator in obs_indicators:
        allDf[indicator] = pd.cut(allDf[indicator], divides, labels=False) / 10
    return allDf

def prehandle(allDf):
    allDf['date'] = pd.to_datetime(allDf['date'])

    # 计算技术指标
    fe = FeatureEngineer(TECHNICAL_INDICATORS_LIST)
    allDf = fe.preprocess_data(allDf)

    tempdate = allDf.date
    allDf.drop("date", axis=1, inplace=True)
    # dropFields(allDf, ['open', 'high', 'close','low','volume', 'amount'])
    allDf = discreteScaler(minMaxScaler(allDf))  # 离散化

    allDf.insert(0, 'date', tempdate)

    return allDf.dropna()



