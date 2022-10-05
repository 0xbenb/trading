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
q = DB_Query_Statement(table_name='rets', columns=['time', 'coin', 'feature', 'value'], filter_col='feature',
                       filter_col_vals=[Y[0], Y[1]])
output_dat = DB_Query(q, db_engine=engine)
output_dat = output_dat.pivot(index=['time', 'coin'], columns='feature', values='value').reset_index()

full_dat = input_dat.merge(output_dat, how='left', on=['time', 'coin'])
full_dat.to_csv('dat/full_dat.csv', index=False)
full_dat = pd.read_csv('dat/full_dat.csv')

# using trailing 1h returns + volume to build a signal to capture changes in price dP and volume dV

# Create plotly charts to ultimately build dashboard
pred_var = X[1]
resp_var = Y[0]

Number_Observations_Time(data=full_dat[['time', 'coin', pred_var]], var_name=pred_var)

skew_t_period = 7 * 24  # start with 1 week - i should find more precisely the natural oscillation frequency here
# might need to detrend the return series

zscore_t_period = 28 * 24

# PREPARING PREDICTOR(S) VARIABLE(S)

# RET_1H_NEUTRAL FOR NOW
full_dat.drop(X[0], axis=1, inplace=True)
# # confirm what this series looks like i.e. detrended - although outliers may affect
px.line(full_dat.groupby('time')[pred_var].mean().reset_index(), x='time', y=pred_var)
# # confirm what this series looks like
px.line(full_dat.groupby('time')[resp_var].mean().reset_index(), x='time', y=resp_var)

# so neutral returns behaving as expected, i wonder what the mean is of non-neutral rets

px.line(full_dat.groupby('time')['fwd_ret_6h'].mean().reset_index(), x='time', y='fwd_ret_6h') # bigger range

full_dat['fwd_ret_6h'].mean() # positive over the entire period. so i wonder how the signal performs if using raw ret
# neutral returns -> 0 which is right. but i wonder when that's happening because i'm using the mean is it being skewed?


full_dat = Calculate_Skew(data=full_dat, variable=pred_var, t_window=skew_t_period, bias=False)
full_dat = Standardise(data=full_dat, method='zscore', variable=f'{pred_var}_skew_7d', t_window=zscore_t_period)
full_dat = Normalise(data=full_dat, variable='ret_1h_neutral_skew_7d_zscore_28d', t_window=zscore_t_period,
                     method='tanh')
full_dat = Remove_Outliers(data=full_dat, GroupBy=['time'], lower_upper_bounds=[2.5, 97.5], variable=resp_var)
full_dat = Remove_Outliers(data=full_dat, GroupBy=['time'], lower_upper_bounds=[2.5, 97.5], variable='fwd_ret_6h')
# All-NaN slice encountered when removing outliers i.e. no data for binance some hours

full_dat.loc[full_dat['time'] == '2018-09-17 04:00:00+00:00']

# more ways to look at this, can do cross sectionally vs peers might be more down the right path let's see

px.histogram(full_dat, x='ret_1h_neutral_skew_7d')
px.histogram(full_dat, x='ret_1h_neutral_skew_7d_zscore_28d') # interesting dual peak when morphing skew to zscore
# maybe the outer regions create opportunity, or perhaps the peaks will be a source of stable/consistent signal
# backtest will reveal

px.scatter(full_dat, x='ret_1h_neutral_skew_7d', y='ret_1h_neutral_skew_7d_zscore_28d')
# signals transforming nicely

px.scatter(full_dat, x='sigmoid', y='tanh')


test_dat = full_dat.sample(100000)

px.scatter(test_dat, x='ret_1h_neutral_skew_7d_zscore_28d_tanh', y='fwd_ret_6h_neutral_rmoutliers',
           marginal_y='histogram', marginal_x='box', trendline='ols', template='plotly_dark')
# R2: ret_1h_neutral_skew_7d_zscore_28d_tanh = 0.000097 slope = -0.0235178

px.scatter(test_dat, x='ret_1h_neutral_skew_7d', y='fwd_ret_6h_neutral_rmoutliers',
           marginal_y='histogram', marginal_x='box', trendline='ols', template='plotly_dark')
# R2: ret_1h_neutral_skew_7d = 0.000147 slope = -0.000119

# see if removing outliers more aggressively to identify trends at this early stage
# should output findings into data table for comparison / learning

# early conclusions
# ret_1h_neutral_skew_7d_zscore_28d_tanh lower R2 more gradient
# ret_1h_neutral_skew_7d higher R2 lesser gradient

# transformation playing a part here

# could standardise normalise return series soon

quicktest = full_dat[['time', 'coin', 'ret_1h_neutral_skew_7d', 'fwd_ret_6h_rmoutliers']]
quicktest.dropna(inplace=True)

quicktest['ret_1h_neutral_skew_7d_bins'] = quicktest.groupby('time')[['ret_1h_neutral_skew_7d']].\
    transform(lambda x: pd.cut(x, bins=5, labels=range(1,6)))

## -- incorrect need to group by time also then average as changes of returns profiles through time will affect
signal_bins = quicktest.groupby(['time', 'ret_1h_neutral_skew_7d_bins'])['fwd_ret_6h_rmoutliers'].\
    median().reset_index()
px.box(signal_bins, x='ret_1h_neutral_skew_7d_bins', y='fwd_ret_6h_rmoutliers')
signal_bins = signal_bins.groupby('ret_1h_neutral_skew_7d_bins')['fwd_ret_6h_rmoutliers'].median().reset_index()
px.bar(signal_bins, x='ret_1h_neutral_skew_7d_bins', y='fwd_ret_6h_rmoutliers')

# SUCCESS - so the distribution is now showing a nice spread of positive to negative trends over the full period
# so the issue here is that when the mean is slightly positive over the full period and then you look at
# returns (neutral) the overall profile will look negative even though the signal is producing a better outcome

full_dat.groupby('time')['fwd_ret_6h_neutral_rmoutliers'].median().mean() # all the fwd_ret_6h_neutral_rmoutlier returns have -ve skew
full_dat.groupby('time')['fwd_ret_6h_neutral'].median().mean() # all the fwd_ret_6h_neutral_rmoutlier returns have -ve skew

# need to pin down where this is coming from

# BACKTEST

# sampling a few monthly examples is a good way to understand
px.bar(quicktest, x='ret_1h_neutral_skew_7d_bins', y='fwd_ret_6h_rmoutliers')


# now to productionise backtesting 

# Loose ends / Reminders
# # factor in 1h constraint for putting on positions




