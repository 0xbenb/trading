import pandas as pd
import direnv
import ccxt
import os
import time
import random
from datetime import datetime, timedelta
from CoreFunctions import *
from EDAFunctions import *
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objects import *
import plotly.io as pio
from scipy.stats import skew
pio.renderers.default = "browser"
direnv.load()
pd.options.display.max_rows = 10
pd.options.display.max_columns = 30
pd.options.display.width = 10000

engine = Create_SQL_Engine()
conn = Create_SQL_Connection(db_engine=engine)

# MARKET PATTERN / OBSERVATION
# markets often quote this mixture of patterns describing them as bullish/bearish movements

# P up V up = BULLISH
# P down V down = BEARISH
# P up V down = BEARISH
# P down V up = BULLISH

# can we combine price / volume trends to predict forward returns

# 7 DAY universe stationary definition of universe - will improve later

univ = Universe_Definition(top100_ndays_ago=7, db_engine=engine)

# PULL IN PREDICTORS FOR TESTING
X = ['volume_1h_usd', 'ret_1h_neutral']

# PREDICTORs
q = DB_Query_Statement(table_name='features', filter_col='feature', filter_col_vals=[X[0]])
volume = DB_Query(q, db_engine=engine)

q = DB_Query_Statement(table_name='rets', filter_col='feature', filter_col_vals=[X[1]])
rets = DB_Query(q, db_engine=engine)

input_dat = pd.concat([volume, rets], axis=0)
input_dat = input_dat.pivot(index=['time', 'coin'], columns='feature', values='value').reset_index()

# PULL IN RESPONSE FOR TESTING
Y = ['fwd_ret_6h_neutral']
q = DB_Query_Statement(table_name='rets', columns=['time', 'coin', 'value'], filter_col='feature',
                       filter_col_vals=[Y[0]])
output_dat = DB_Query(q, db_engine=engine)
output_dat.rename({'value': 'fwd_ret'}, axis=1, inplace=True)

full_dat = input_dat.merge(output_dat, how='left', on=['time', 'coin'])
# factor in 1h constraint for putting on positions

# using trailing 1h returns + volume to build a signal to capture changes in price dP and volume dV

# Create plotly charts to ultimately build dashboard
data = full_dat.copy(deep=True)

pred_var = X[1]
resp_var = Y[0]
data = data[['time', 'coin', pred_var]]

Number_Observations_Time(data=full_dat[['time', 'coin', pred_var]], var_name=pred_var)

# massive tails in rets i need to clean this
test_dat = full_dat.sample(100000)
px.scatter(test_dat, x='volume_1h_usd', y='fwd_ret', marginal_y='histogram', marginal_x='box', trendline='ols',
           template='plotly_dark')


x =[1,2,3,4,5,5,5,10]
print(skew(x))
print(skew(x, bias=False))  # option to adjust for statistical bias - need to understand this more

# how do i want to capture changes in price & volume?
# many ways to categorise i'm going to try a few and do a quick test of which ones seem more promising

# z-score of price and volumes
# here i'm leaning towards using skew to represent the movement as i want to not only see spikes in volumes
# i also want to understand if the overall profile is looking skewed vs history compared to other coins

# 1. calculate skew of returns per coin over trailing period
# 2. this will capture coins with +ve skew (-ve performance) -ve skew (positive performance)
# 3. coins with big skew with a jump will be prime for reversal

# might be doing too much here could do more simply but good direction of travel












