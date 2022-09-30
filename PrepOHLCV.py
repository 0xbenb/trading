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

# 7 DAY universe stationary definition of universe - will improve later
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

univ = Universe_Definition(top100_ndays_ago=7)

# PULL PRICE AND VOLUME DATA

q = DB_Query_Statement(table_name='binance_ohlcv', columns=['time','symbol','o','volume'])
ohlcv = DB_Query(query=q, db_engine=engine).dropna()
ohlcv['time'] = pd.to_datetime(ohlcv['time'], format='%Y-%m-%d %H:%M:%S')
ohlcv.rename({'o': 'price'}, axis=1, inplace=True)

# PREPARE DATA TO CALCULATE RETURNS AND VOLUME IN USD

def Calculate_USD_Price_Volume(ohlcv_dat, univ_dat):


    ohlcv_dat = ohlcv_dat.merge(univ[['symbol','base','quote']], on='symbol', how='left')

    quote_symbols = ['BTC/USDT','ETH/USDT']
    quote_prices = ohlcv_dat.copy(deep=True)
    quote_prices = quote_prices[quote_prices['symbol'].isin(quote_symbols)]
    quote_prices.drop(['quote','symbol','volume'], axis=1, inplace=True)
    quote_prices.rename({'price': 'quote_price_usd', 'base': 'quote'}, axis=1, inplace=True)

    ohlcv_dat = ohlcv_dat.merge(quote_prices, how='left', on=['time','quote'])
    ohlcv_dat['quote_price_usd'] = np.where(ohlcv_dat['quote'] == 'USDT', 1, ohlcv_dat['quote_price_usd'])  # usdt quote price rate = 1
    ohlcv_dat.dropna(inplace=True) # btcusdt didn't exist on platform before august 2017
    ohlcv_dat['price_usd'] = ohlcv_dat['price'] * ohlcv_dat['quote_price_usd']
    ohlcv_dat['volume_1h_usd'] = ohlcv_dat['volume'] * ohlcv_dat['price_usd']

    # here i am assuming the # venues doesn't matter i'm just adding the volume for each base
    ohlcv_smy = ohlcv_dat.groupby(['time','base']).agg(
        {'price_usd': 'mean', 'volume_1h_usd': 'sum'}
    ).reset_index().rename({'base': 'coin'}, axis=1)
    
    return ohlcv_smy

ohlcv_smy[(ohlcv_smy['time'] >= '2019-11-13 01:00:00+00:00') & (ohlcv_smy['coin'] == 'ADA')]

all_coins = ohlcv_smy['coin'].unique()
all_dates = pd.date_range(start=ohlcv_smy['time'].min(), end=ohlcv_smy['time'].max(), freq='h')

complete_index = list(itertools.product(all_dates, all_coins))
complete_index = pd.DataFrame(complete_index, columns=['time', 'coin'])

min_dt = ohlcv_smy.groupby('coin')['time'].min().reset_index().rename({'time': 'min_dt'}, axis=1)
complete_index = complete_index.merge(min_dt, how='left', on='coin')
complete_index = complete_index[complete_index['time'] >= complete_index['min_dt']]
complete_index.drop('min_dt', axis=1, inplace=True)

ohlcv_smy = complete_index.merge(ohlcv_smy, how='left', on=['time','coin'])

ohlcv_smy['shift_price_usd'] = ohlcv_smy.groupby('coin')['price_usd'].shift(1)
# ohlcv_smy['shift'] = ohlcv_smy.groupby('coin')['time'].shift(-1)
# ohlcv_smy['diff'] = ohlcv_smy['shift'] - ohlcv_smy['time']


# create perfect index then join


# RANGE OF FORWARD RETURN TIME PERIODS
def FWD_Return(time_period):


# RANGE OF NEUTRAL FORWARD RETURN TIME PERIODS

# RANGE OF TRAILING RETURN TIME PERIODS

# RANGE OF





























