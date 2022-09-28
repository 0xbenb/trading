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

ts_7d_ago = str(datetime.utcnow() - timedelta(days=7)) # reduce the noise of coins entering / leaving universe

q = DB_Query_Statement(table_name='universe', columns = ['binance_symbol','binance_base','binance_quote'],
                       time_start=ts_7d_ago)
univ = DB_Query(query=q, db_engine=engine)
univ.columns = univ.columns.str.removeprefix("binance_")
univ = univ.dropna().drop_duplicates()

# one problem with univ is that you start to rule out coins that get delisted i.e. survivorship bias creeping in
# i need to change univ to have a column saying "binance_delisted"
# that way when i drag in data i can get full picture not more perfect universe

# have pricing in USDT BTC ETH
# will standardise to $ pricing (USDT as proxy)

q = DB_Query_Statement(table_name='binance_ohlcv', columns=['time','symbol','c'])
# ohlcv = DB_Query(query=q, db_engine=engine).dropna()
# ohlcv.to_csv('dat/ohlcv.csv', index=False)
ohlcv = pd.read_csv('dat/ohlcv.csv')
ohlcv.rename({'c': 'price'}, axis=1, inplace=True)

# sample_symbols = list(np.random.choice(ohlcv['symbol'].unique(), 10))
# sample_symbols = sample_symbols + ['BTC/USDT','ETH/USDT','BNB/ETH']
# ohlcv = ohlcv[ohlcv['symbol'].isin(sample_symbols)]

ohlcv = ohlcv.merge(univ[['symbol','base','quote']], on='symbol', how='left')

# for each symbol bring on $ price for each quote to be able to calculate everything in dollars
quote_symbols = ['BTC/USDT','ETH/USDT']
quote_prices = ohlcv[ohlcv['symbol'].isin(quote_symbols)]
quote_prices.drop(['quote','symbol'], axis=1, inplace=True)
quote_prices.rename({'price': 'quote_price_usd', 'base': 'quote'}, axis=1, inplace=True)

ohlcv = ohlcv.merge(quote_prices, how='left', on=['time','quote'])

# where the quote is already usdt set the quote price rate = 1
ohlcv['quote_price_usd'] = np.where(ohlcv['quote'] == 'USDT', 1, ohlcv['quote_price_usd'])

ohlcv.dropna(inplace=True)

ohlcv['price_usd'] = ohlcv['price'] * ohlcv['quote_price_usd']

price_usd = ohlcv.groupby(['time','base'])['price_usd'].mean().reset_index()

# data checks
ohlcv[ohlcv['time'] == '2022-09-28 14:00:00+00:00'].sample(10)






























