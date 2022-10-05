import pandas as pd
import direnv
import ccxt
import os
import time
import random
from datetime import datetime, timedelta
from CoreFunctions import *
direnv.load()
pd.options.display.max_rows = 10
pd.options.display.max_columns = 30
pd.options.display.width = 10000

engine = Create_SQL_Engine()
conn = Create_SQL_Connection(db_engine=engine)

def Calculate_USD_Price_Volume(ohlcv_dat, univ_dat):

    ohlcv_dat = ohlcv_dat.merge(univ_dat[['symbol', 'base', 'quote']], on='symbol', how='left')

    quote_symbols = ['BTC/USDT', 'ETH/USDT']
    quote_prices = ohlcv_dat.copy(deep=True)
    quote_prices = quote_prices[quote_prices['symbol'].isin(quote_symbols)]
    quote_prices.drop(['quote', 'symbol', 'volume'], axis=1, inplace=True)
    quote_prices.rename({'price': 'quote_price_usd', 'base': 'quote'}, axis=1, inplace=True)

    ohlcv_dat = ohlcv_dat.merge(quote_prices, how='left', on=['time', 'quote'])
    ohlcv_dat['quote_price_usd'] = np.where(ohlcv_dat['quote'] == 'USDT', 1,
                                            ohlcv_dat['quote_price_usd'])  # usdt quote price rate = 1
    ohlcv_dat.dropna(inplace=True)  # btcusdt didn't exist on platform before august 2017
    ohlcv_dat['price_usd'] = ohlcv_dat['price'] * ohlcv_dat['quote_price_usd']
    ohlcv_dat['volume_1h_usd'] = ohlcv_dat['volume'] * ohlcv_dat['price_usd']

    # here i am assuming the # venues doesn't matter i'm just adding the volume for each base
    output = ohlcv_dat.groupby(['time', 'base']).agg(
        {'price_usd': 'mean', 'volume_1h_usd': 'sum'}
    ).reset_index().rename({'base': 'coin'}, axis=1)

    return output


def Create_Perfect_Index(imperfect_dat: pd.DataFrame):
    """

    :param imperfect_dat: for now only built to assume index of time + coin
    :return:
    """
    all_coins = imperfect_dat['coin'].unique()
    all_dates = pd.date_range(start=imperfect_dat['time'].min(), end=imperfect_dat['time'].max(), freq='h')

    complete_index = list(itertools.product(all_dates, all_coins))
    complete_index = pd.DataFrame(complete_index, columns=['time', 'coin'])

    min_dt = imperfect_dat.groupby('coin')['time'].min().reset_index().rename({'time': 'min_dt'}, axis=1)
    complete_index = complete_index.merge(min_dt, how='left', on='coin')
    complete_index = complete_index[complete_index['time'] >= complete_index['min_dt']]
    complete_index.drop('min_dt', axis=1, inplace=True)

    complete_dat = complete_index.merge(imperfect_dat, how='left', on=['time','coin'])

    return complete_dat


def Calculate_Returns(price_data: pd.DataFrame, time_period: int, price_name: str = None):
    """

    :param price_data: data block of [time,coin,price]
    :param time_period: time period in hours 1h, 3h, 6h, 12h, 1d, 3d, 7d (+ve = FWD -ve = trailing)
    :param price_name: name of price columns if 'price' not name
    :return:
    """
    if price_name:
        price_data.rename({price_name: 'price'}, axis=1, inplace=True)

    # static options for now
    time_period_names = {1: '1h', 3: '3h', 6: '6h', 12: '12h', 24: '1d', 72: '3d', 168: '7d'}

    price_data = price_data[['time', 'coin', 'price']]

    price_data['shift_price'] = price_data.groupby('coin')['price'].shift(-time_period)  #

    ret_name = f'ret_{time_period_names[abs(time_period)]}'

    if time_period > 0:
        ret_name = f'fwd_{ret_name}'
        price_data[ret_name] = price_data['shift_price'] / price_data['price'] - 1

    if time_period < 0:
        price_data[ret_name] = price_data['price'] / price_data['shift_price'] - 1

    price_data.drop(['price','shift_price'], axis=1, inplace=True)

    mkt_ret_name = f'mkt_ret_{time_period}h'

    price_data[mkt_ret_name] = price_data.groupby('time')[ret_name].transform('median')

    price_data[f'{ret_name}_neutral'] = price_data[ret_name] - price_data[mkt_ret_name]

    price_data.drop(mkt_ret_name, axis=1, inplace=True)

    price_data = price_data.set_index(['time', 'coin'])

    return price_data






