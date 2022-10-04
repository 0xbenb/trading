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
output_dat.rename({'value': Y[0]}, axis=1, inplace=True)

full_dat = input_dat.merge(output_dat, how='left', on=['time', 'coin'])
full_dat.to_csv('dat/full_dat.csv', index=False)
full_dat = pd.read_csv('dat/full_dat.csv')

# using trailing 1h returns + volume to build a signal to capture changes in price dP and volume dV

# Create plotly charts to ultimately build dashboard
pred_var = X[1]
resp_var = Y[0]

Number_Observations_Time(data=full_dat[['time', 'coin', pred_var]], var_name=pred_var)

full_dat = pd.read_csv('dat/full_dat.csv')
# full_dat = full_dat[['time', 'coin', pred_var]]
skew_t_period = 7 * 24  # start with 1 week
zscore_t_period = 28 * 24

# PREPARING PREDICTOR(S) VARIABLE(S)

# in terms of treating outliers, don't want to over clean. generally raw data > calculate > standardise > normalise
# can ultimately test approach to see which option gives the best predictions

# TRAILING RETURNS (CHANGE IN PRICE)

# CLEAN RETURNS
# remove when doing backtesting?

# VOLUME (CHANGE IN VOLUME)
#   ZSCORE I.E. STANDARDISE (NEUTRAL / ZSCORE) - MAYBE JUST ZSCORE IS ENOUGH
#   SKEW

# work through just prep of 'ret_1h_neutral' before repeating similar for volume
full_dat = Calculate_Skew(data=full_dat, variable=pred_var, t_window=skew_t_period, bias=False)
full_dat = Standardise(data=full_dat, method='zscore', variable=f'{pred_var}_skew_7d', t_window=zscore_t_period)
# more ways to look at this, can do cross sectionally vs peers

px.histogram(full_dat, x='ret_1h_neutral_skew_7d')
px.histogram(full_dat, x='ret_1h_neutral_skew_7d_zscore_28d') # interesting dual peak when morphing skew to zscore
# maybe the outer regions create opportunity, or perhaps the peaks will be a source of stable/consistent signal
# backtest will reveal

px.scatter(full_dat, x='ret_1h_neutral_skew_7d', y='ret_1h_neutral_skew_7d_zscore_28d')
# signals transforming nicely

# will reduce the extremes of the values by normalising to make data more usable for modelling via a transform fn
full_dat = Normalise(data=full_dat, variable='ret_1h_neutral_skew_7d_zscore_28d', t_window=zscore_t_period,
                     method='tanh')

px.scatter(full_dat, x='sigmoid', y='tanh')


# FINISH PREPARING RESPONSE VARIABLE
# for initial quicktest only going to remove outliers
# will prepare the raw return series more thoroughly for testing relationship statistically

# CLEAN RETURNS (REMOVE TOP / BOTTOM 1-2.5%)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# remove top / bottom 2.5% (per time)
full_dat = Remove_Outliers(data=full_dat, GroupBy=['time'], lower_upper_bounds=[2.5, 97.5], variable=resp_var)

test_dat = full_dat.sample(100000)

px.scatter(test_dat, x='ret_1h_neutral_skew_7d_zscore_28d_tanh', y='fwd_ret_6h_neutral_rmoutliers',
           marginal_y='histogram', marginal_x='box', trendline='ols', template='plotly_dark')
# R2: ret_1h_neutral_skew_7d_zscore_28d_tanh = 0.000097 slope = -0.0235178

px.scatter(test_dat, x='ret_1h_neutral_skew_7d', y='fwd_ret_6h_neutral_rmoutliers',
           marginal_y='histogram', marginal_x='box', trendline='ols', template='plotly_dark')
# R2: ret_1h_neutral_skew_7d = 0.000147 slope = -0.000119


# see if removing outliers more aggressively to identify trends at this early stage
# should output findings into data table for comparison / learning

# very low R2 (rm outliers might show more of a trend)
# ret_1h_neutral_skew_7d has better R2 but lower trend

full_dat = Remove_Outliers(data=full_dat, GroupBy=['time'], lower_upper_bounds=[10, 90], variable=resp_var)
test_dat = full_dat.sample(100000)
px.scatter(test_dat, x='ret_1h_neutral_skew_7d_zscore_28d_tanh', y='fwd_ret_6h_neutral_rmoutliers',
           marginal_y='histogram', marginal_x='box', trendline='ols', template='plotly_dark')
# R2: ret_1h_neutral_skew_7d_zscore_28d_tanh = 0.000093 slope = -0.018766

px.scatter(test_dat, x='ret_1h_neutral_skew_7d', y='fwd_ret_6h_neutral_rmoutliers',
           marginal_y='histogram', marginal_x='box', trendline='ols', template='plotly_dark')
# R2: ret_1h_neutral_skew_7d = 0.000401 slope = -0.000147

# early conclusions
# ret_1h_neutral_skew_7d_zscore_28d_tanh lower R2 more gradient
# ret_1h_neutral_skew_7d higher R2 lesser gradient

# transformation playing a part here

# should standardise return series

quicktest = full_dat[['time', 'coin', 'ret_1h_neutral_skew_7d', 'fwd_ret_6h_neutral_rmoutliers']]
quicktest.dropna(inplace=True)

quicktest['ret_1h_neutral_skew_7d_bins'] = quicktest.groupby('time')[['ret_1h_neutral_skew_7d']].\
    transform(lambda x: pd.cut(x, bins=5, labels=range(1,6)))

signal_bins = quicktest.groupby('ret_1h_neutral_skew_7d_bins')['fwd_ret_6h_neutral_rmoutliers'].mean().reset_index()
px.bar(signal_bins, x='ret_1h_neutral_skew_7d_bins', y='fwd_ret_6h_neutral_rmoutliers')
# maybe something there

signal_bins = quicktest.groupby('ret_1h_neutral_skew_7d_bins')['fwd_ret_6h_neutral_rmoutliers'].median().reset_index()
px.bar(signal_bins, x='ret_1h_neutral_skew_7d_bins', y='fwd_ret_6h_neutral_rmoutliers')
# less so but still meaningful difference between the tails where it matters i think for this signal


quicktest = full_dat[['time', 'coin', 'ret_1h_neutral_skew_7d_zscore_28d_tanh', 'fwd_ret_6h_neutral_rmoutliers']]
quicktest.dropna(inplace=True)

quicktest['ret_1h_neutral_skew_7d_zscore_28d_tanh_bins'] = quicktest.groupby('time')[['ret_1h_neutral_skew_7d_zscore_28d_tanh']].\
    transform(lambda x: pd.cut(x, bins=5, labels=range(1,6)))

signal_bins = quicktest.groupby('ret_1h_neutral_skew_7d_zscore_28d_tanh_bins')['fwd_ret_6h_neutral_rmoutliers'].mean().reset_index()
px.bar(signal_bins, x='ret_1h_neutral_skew_7d_zscore_28d_tanh_bins', y='fwd_ret_6h_neutral_rmoutliers')
# maybe something there

signal_bins = quicktest.groupby('ret_1h_neutral_skew_7d_zscore_28d_tanh_bins')['fwd_ret_6h_neutral_rmoutliers'].median().reset_index()
px.bar(signal_bins, x='ret_1h_neutral_skew_7d_zscore_28d_tanh_bins', y='fwd_ret_6h_neutral_rmoutliers')

# this quicktest aligns with previous findings about R2 for the ret_1h_neutral_skew_7d compared to the
# ret_1h_neutral_skew_7d_zscore_28d_tanh_bins with additional z score + transformation (though transformation won't affect this test)
# so live skew, not looking at changes (zscore) over a trailing period seems more informative

# i think interesting way to test progression of this would be combining ret_1h_neutral_skew_7d (zscore) with
# ret_1h_neutral_skew_zscore_Xd over a time period X

# outstanding question -- why are all the neutral returns negative? wouldn't have thought this to be the case

# without additional testing / developing i would bring forward ret_1h_neutral_skew_7d to backtest stage



# BACKTEST







# Loose ends / Reminders
# # factor in 1h constraint for putting on positions




