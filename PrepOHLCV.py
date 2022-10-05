import pandas as pd
import direnv
import ccxt
import os
import time
import random
from datetime import datetime, timedelta
from CoreFunctions import *
from PrepFunctions import *
direnv.load()
pd.options.display.max_rows = 10
pd.options.display.max_columns = 30
pd.options.display.width = 10000

engine = Create_SQL_Engine()
conn = Create_SQL_Connection(db_engine=engine)

# 7 DAY universe stationary definition of universe - will improve later

univ = Universe_Definition(top100_ndays_ago=7, db_engine=engine)

###########################
# PULL DATA FROM DATABASE #
###########################

q = DB_Query_Statement(table_name='binance_ohlcv', columns=['time', 'symbol', 'o', 'volume'])
ohlcv = DB_Query(query=q, db_engine=engine).dropna()
ohlcv['time'] = pd.to_datetime(ohlcv['time'], format='%Y-%m-%d %H:%M:%S')
ohlcv.rename({'o': 'price'}, axis=1, inplace=True)

#######################################################
# PREPARE DATA TO CALCULATE RETURNS AND VOLUME IN USD #
#######################################################

ohlcv_smy = Calculate_USD_Price_Volume(ohlcv_dat=ohlcv, univ_dat=univ)

ohlcv_smy = Create_Perfect_Index(imperfect_dat=ohlcv_smy)

########################################################
# RANGE OF TIME PERIODS FORWARD/TRAILING RETURNS/NEUTRAL
########################################################

ret_periods = [1, -1, 3, -3, 6, -6, 12, -12, 24, -24, 72, -72, 168, -168]
all_returns = [Calculate_Returns(price_data=ohlcv_smy, time_period=i, price_name='price_usd') for i in ret_periods]

all_returns = pd.concat(all_returns, axis=1).reset_index()

all_returns = pd.melt(all_returns, id_vars=['time', 'coin'], var_name='feature')

ohlcv_smy.rename({'price': 'price_usd'}, axis=1, inplace=True)
ohlcv_smy = pd.melt(ohlcv_smy, id_vars=['time', 'coin'], var_name='feature')

###########################
# IMPORT DATA TO DATABASE #
###########################

import_dat = pd.concat([all_returns, ohlcv_smy[ohlcv_smy['feature'] == 'price_usd']], axis=0)

Create_Database_Table(table_name='rets', db_engine=engine, db_conn=conn)

data_splits = Data_Splitter(import_dat, max_rows=10**6)

# RETURNS
for dat in data_splits:
    pop(data=dat, table_name='rets', db_engine=engine)

t0 = time.time()
q = "select distinct time, coin from rets"
dat = pd.read_sql_query(q, con=engine)
print(time.time() - t0)
# 39 second query - inefficient way, will do for now can store this output more efficiently later
# would be good to store this info of t_start t_end into database (same for ohlcv)
# stored proc for postgres?

# VOLUMES
ohlcv_smy = ohlcv_smy[ohlcv_smy['feature'] == 'volume_1h_usd']

Create_Database_Table(table_name='features', db_engine=engine, db_conn=conn)

pop(data=ohlcv_smy, table_name='features', db_engine=engine)








