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

univ = Universe_Definition(top100_ndays_ago=7, db_engine=engine)

# PULL IN FEATURES FOR TESTING

# keep simple to start -> returns and volume

q = DB_Query_Statement(table_name='features', filter_col='feature', filter_col_vals=['volume_1h_usd'])
volume = DB_Query(q, db_engine=engine)

q = DB_Query_Statement(table_name='rets', filter_col='feature', filter_col_vals=['ret_1h_neutral'])
rets = DB_Query(q, db_engine=engine)

# using trailing 1h returns + volume to build a signal to capture changes in price dP and volume dV

# MARKET PATTERN / OBSERVATION
# markets often quote this mixture of patterns describing them as bullish/bearish movements

# P up V up = BULLISH
# P down V down = BEARISH
# P up V down = BEARISH
# P down V up = BULLISH

# can we combine price / volume trends to predict forward returns














