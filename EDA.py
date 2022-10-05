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
pio.renderers.default = "browser"
from scipy.stats import skew
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

direnv.load()
pd.options.display.max_rows = 10
pd.options.display.max_columns = 30
pd.options.display.width = 10000

engine = Create_SQL_Engine()
conn = Create_SQL_Connection(db_engine=engine)

X = ['volume_1h_usd', 'ret_1h_neutral']
Y = ['fwd_ret_6h_neutral', 'fwd_ret_6h']

###########################
#  INTRO RESEARCH TOPIC #
###########################

# MARKET PATTERN / OBSERVATION
# markets often quote this mixture of patterns describing them as bullish/bearish movements

# P up V up = BULLISH
# P down V down = BEARISH
# P up V down = BEARISH
# P down V up = BULLISH

# can we combine price / volume trends to predict forward returns

# how do i want to capture changes in price & volume?
# many ways to categorise i'm going to try a few and do a quick test of which ones seem more promising

# z-score of price and volumes
# i'm leaning towards using skew to represent the movement as i want to not only see spikes in volumes
# i also want to understand if the overall profile is looking skewed vs history compared to other coins

# 1. calculate skew of returns per coin over trailing period
# 2. this will capture coins with +ve skew (-ve performance) -ve skew (positive performance)
# 3. coins with big skew with a jump will be prime for reversal

# might be doing too much here could do more simply but good direction of travel
# two kind of ideas here histkew & relskew & probably a combo is interesting or almost a diffskew something


###########################
# PULL DATA FROM DATABASE #
###########################

# UNIVERSE
univ = Universe_Definition(top100_ndays_ago=7, db_engine=engine)

# PREDICTOR VARIABLES
q = DB_Query_Statement(table_name='features', filter_col='feature', filter_col_vals=[X[0]])
volume = DB_Query(q, db_engine=engine)

q = DB_Query_Statement(table_name='rets', filter_col='feature', filter_col_vals=[X[1]])
rets = DB_Query(q, db_engine=engine)

# COMBINE PREDICTORS
input_dat = pd.concat([volume, rets], axis=0)
input_dat = input_dat.pivot(index=['time', 'coin'], columns='feature', values='value').reset_index()

# RESPONSE VARIABLES
q = DB_Query_Statement(table_name='rets', columns=['time', 'coin', 'feature', 'value'], filter_col='feature',
                       filter_col_vals=[Y[0], Y[1]])
output_dat = DB_Query(q, db_engine=engine)
output_dat = output_dat.pivot(index=['time', 'coin'], columns='feature', values='value').reset_index()

full_dat = input_dat.merge(output_dat, how='left', on=['time', 'coin'])
# full_dat.to_csv('dat/full_dat.csv', index=False)

###########################
#    ASSIGN VARIABLES     #
###########################

# CAN LOOP THROUGH VARIATIONS
pred_var = X[1]
resp_var = Y[0]
skew_t_period = 7 * 24
zscore_t_period = 28 * 24

Number_Observations_Time(data=full_dat[['time', 'coin', pred_var]], var_name=pred_var)

###############################
# PREPARE PREDICTOR VARIABLES #
###############################

full_dat = pd.read_csv('dat/full_dat.csv')
full_dat['fwd_ret_6h_neutral'].mean() # this is very close to 0

# CALCULATE SKEW
full_dat = Calculate_Skew(data=full_dat, variable=pred_var, t_window=skew_t_period, bias=False)
# STANDARDISE
full_dat = Standardise(data=full_dat, method='zscore', variable=f'{pred_var}_skew_7d', t_window=zscore_t_period)
# NORMALISE
full_dat = Normalise(data=full_dat, variable='ret_1h_neutral_skew_7d_zscore_28d', t_window=zscore_t_period,
                     method='tanh')
# REMOVE OUTLIERS
full_dat = Remove_Outliers(data=full_dat, GroupBy=['time'], lower_upper_bounds=[2.5, 97.5], variable=resp_var)
full_dat = Remove_Outliers(data=full_dat, GroupBy=['time'], lower_upper_bounds=[2.5, 97.5], variable='fwd_ret_6h')
# SIGNAL BINS
full_dat = Create_Bins(data=full_dat, GroupBy=['time'], variable='ret_1h_neutral_skew_7d')



# Loose ends / Reminders
# # factor in 1h constraint for putting on positions

# px.scatter(test_dat, x='ret_1h_neutral_skew_7d_zscore_28d_tanh', y='fwd_ret_6h_neutral_rmoutliers',
#            marginal_y='histogram', marginal_x='box', trendline='ols', template='plotly_dark')



