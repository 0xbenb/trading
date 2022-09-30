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


def Universe_Definition(top100_ndays_ago: int):
    """

    :param top100_ndays_ago: how many days ago in the top 100 coingecko
    :return:
    """
    ts_nd_ago = str(datetime.utcnow() - timedelta(days=top100_ndays_ago)) # reduce the noise of coins entering / leaving universe
    sql_q = DB_Query_Statement(table_name='universe', columns=['binance_symbol','binance_base','binance_quote'],
                               time_start=ts_nd_ago)
    univ_nd = DB_Query(query=sql_q, db_engine=engine)
    univ_nd.columns = univ_nd.columns.str.removeprefix("binance_")
    univ_nd = univ_nd.dropna().drop_duplicates()

    return univ_nd


def Calculate_USD_Price_Volume(ohlcv_dat, univ_dat):
    ohlcv_dat = ohlcv_dat.merge(univ[['symbol', 'base', 'quote']], on='symbol', how='left')

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