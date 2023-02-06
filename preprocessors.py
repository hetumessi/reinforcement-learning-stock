import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf


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



