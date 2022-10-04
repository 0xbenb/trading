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

X = ['volume_1h_usd', 'ret_1h_neutral']
Y = ['fwd_ret_6h_neutral']

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


# 7 DAY universe stationary definition of universe - will improve later

univ = Universe_Definition(top100_ndays_ago=7, db_engine=engine)

# PULL IN PREDICTORS FOR TESTING

# PREDICTORs
q = DB_Query_Statement(table_name='features', filter_col='feature', filter_col_vals=[X[0]])
volume = DB_Query(q, db_engine=engine)

q = DB_Query_Statement(table_name='rets', filter_col='feature', filter_col_vals=[X[1]])
rets = DB_Query(q, db_engine=engine)

input_dat = pd.concat([volume, rets], axis=0)
input_dat = input_dat.pivot(index=['time', 'coin'], columns='feature', values='value').reset_index()

# PULL IN RESPONSE FOR TESTING
q = DB_Query_Statement(table_name='rets', columns=['time', 'coin', 'value'], filter_col='feature',
                       filter_col_vals=[Y[0]])
output_dat = DB_Query(q, db_engine=engine)
output_dat.rename({'value': 'fwd_ret'}, axis=1, inplace=True)

full_dat = input_dat.merge(output_dat, how='left', on=['time', 'coin'])
full_dat.to_csv('dat/full_dat.csv', index=False)
full_dat = pd.read_csv('dat/full_dat.csv')

# using trailing 1h returns + volume to build a signal to capture changes in price dP and volume dV

# Create plotly charts to ultimately build dashboard
pred_var = X[1]
resp_var = Y[0]

Number_Observations_Time(data=full_dat[['time', 'coin', pred_var]], var_name=pred_var)

# graph idea: massive tails in rets i need to clean this
test_dat = full_dat.sample(100000)
px.scatter(test_dat, x='volume_1h_usd', y='fwd_ret', marginal_y='histogram', marginal_x='box', trendline='ols',
           template='plotly_dark')


full_dat = pd.read_csv('dat/full_dat.csv')
full_dat = full_dat[['time', 'coin', pred_var]]
t_period = 7 * 24  # start with 1 week


# pandas calculates unbiased skew i.e. bias = False (scipy defaults to true)
# If bias = False, the calculations are corrected for bias
# To make scipy.stats.skew compute the same value as the skew() method in Pandas, add the argument bias=False.
# scipy calculates biased skew as default i.e. bias = True

full_dat = Calculate_Skew(data=full_dat, variable=pred_var, t_window=t_period, bias=False)

full_dat[f'{pred_var}_skew'] = full_dat.groupby('coin')[pred_var].rolling(t_period).skew().reset_index(level=0, drop=True)

full_dat = Standardise(data=full_dat, method='zscore', t_window=t_period, variable=f'{pred_var}_skew')

# PREPARING PREDICTOR(S) VARIABLE(S)

# in terms of treating outliers, don't want to over clean. generally raw data > calculate > standardise > normalise
# can ultimately test approach to see which option gives the best predictions

# TRAILING RETURNS (CHANGE IN PRICE)

# CLEAN RETURNS
# REMOVE TOP / BOTTOM 1-2.5%)
# - normalise
# - https://stackoverflow.com/questions/29275210/how-to-use-sigmoid-function-to-normalized-a-list-of-vaues
# MAD
# SIGMOID -1 & +1

# https://stats.stackexchange.com/questions/7757/data-normalization-and-standardization-in-neural-networks
# import sklearn
#
# # Normalize X, shape (n_samples, n_features)
# X_norm = sklearn.preprocessing.normalize(X)

# https://stackoverflow.com/questions/43061120/tanh-estimator-normalization-in-python

unnormalizedData = np.array([1,5,3,12,1,4345,4], dtype=np.float64)

m = np.mean(unnormalizedData, axis=0) # array([16.25, 26.25])
std = np.std(unnormalizedData, axis=0) # array([17.45530005, 22.18529919])

data = 0.5 * (np.tanh(0.01 * ((unnormalizedData - m) / std)) + 1)

np.tanh(data)

full_dat['ret_1h_neutral_tanh'] = np.tan(full_dat[pred_var])

# normalise --> scale
# i.e. calculate z score
# apply sigmoid function
# done

# https://www.geeksforgeeks.org/python-pytorch-tanh-method/
# https://www.geeksforgeeks.org/python-tensorflow-nn-tanh/


#   ZSCORE (ALREADY NEUTRAL)
#   SKEW

# VOLUME (CHANGE IN VOLUME)
#   ZSCORE I.E. STANDARDISE (NEUTRAL / ZSCORE) - MAYBE JUST ZSCORE IS ENOUGH
#   SKEW

# FINSIH PREPARING RESPONSE VARIABLE
# CLEAN RETURNS (REMOVE TOP / BOTTOM 1-2.5%)
# APPLY SIGMOID FUNCTION

# SPLIT INTO BUCKETS LOOK AT NEUTRAL RET PROFILE
# DECIDE HOW TO EVOLVE / COMBINE SIGNALS

# BACKTEST







# Loose ends / Reminders
# # factor in 1h constraint for putting on positions




