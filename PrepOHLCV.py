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

univ = Universe_Definition(top100_ndays_ago=7)

# PULL PRICE AND VOLUME DATA

q = DB_Query_Statement(table_name='binance_ohlcv', columns=['time','symbol','o','volume'])
ohlcv = DB_Query(query=q, db_engine=engine).dropna()
ohlcv['time'] = pd.to_datetime(ohlcv['time'], format='%Y-%m-%d %H:%M:%S')
ohlcv.rename({'o': 'price'}, axis=1, inplace=True)

# PREPARE DATA TO CALCULATE RETURNS AND VOLUME IN USD

ohlcv_smy = Calculate_USD_Price_Volume(ohlcv_dat=ohlcv, univ_dat=univ)

ohlcv_smy = Create_Perfect_Index(imperfect_dat=ohlcv_smy)


# RANGE OF FORWARD RETURN TIME PERIODS
def FWD_Return(time_period):
    

# RANGE OF NEUTRAL FORWARD RETURN TIME PERIODS

# RANGE OF TRAILING RETURN TIME PERIODS

# RANGE OF





























