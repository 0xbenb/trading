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
# factor in 1h constraint for putting on positions

# using trailing 1h returns + volume to build a signal to capture changes in price dP and volume dV

# Create plotly charts to ultimately build dashboard
pred_var = X[1]
resp_var = Y[0]

Number_Observations_Time(data=full_dat[['time', 'coin', pred_var]], var_name=pred_var)

# graph idea: massive tails in rets i need to clean this
test_dat = full_dat.sample(100000)
px.scatter(test_dat, x='volume_1h_usd', y='fwd_ret', marginal_y='histogram', marginal_x='box', trendline='ols',
           template='plotly_dark')


x =[1,2,3,4,5,5,5,10]
print(skew(x))
print(skew(x, bias=False))  # option to adjust for statistical bias - need to understand this more

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

# start simple let's just build skew first

full_dat = pd.read_csv('dat/full_dat.csv')
full_dat = full_dat[['time', 'coin', pred_var]]
t_period = 14 * 24 #  14 days

def Sample_Data(data:pd.DataFrame, column: str, n_sample):
    subset = np.random.choice(data[column].unique(), n_sample)
    data = data[data[column].isin(subset)]

    return data

def calc_skew(x):
    return skew(x, bias=False)  # calculations are corrected for bias is bias=False

sample_dat = Sample_Data(data=full_dat, column='coin', n_sample=5)

res = sample_dat.groupby('coin', group_keys=True)['ret_1h_neutral'].apply(
    lambda x: x.rolling(5).apply(calc_skew)
)
sample_dat['skew'] = res.reset_index(level=0, drop=True)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

z = sample_dat.loc[(full_dat['coin'] == 'VET') & (full_dat['time'] >= '2018-07-25 05:00:00+00:00')]

z['ret_1h_neutral'].iloc[0:5]
# https://stackoverflow.com/questions/51935456/std-skew-giving-wrong-answer-with-rolling

# pandas calculates unbiased skew i.e. bias = False (scipy defaults to true)
# If bias = False, the calculations are corrected for bias
# To make scipy.stats.skew compute the same value as the skew() method in Pandas, add the argument bias=False.
# scipy calculates biased skew as default i.e. bias = True


sample_dat['pandas_skew'] = sample_dat.groupby('coin')[pred_var].rolling(5).skew().reset_index(level=0, drop=True)
# shows that bias = False with pandas i.e. it is unbiased value

sample_dat['test'] = round(sample_dat['skew'], 6) == round(sample_dat['pandas_skew'], 6) # the same apart from rounding



# next need to think about how to prepare the other input
# volume 1h usd
# neutral volume?
# you can z-score per coin rolling window which will pick up changes i suppose
# i wonder if doing % volume change (- market volume) is a good way to take this


# formulate testing process strategy thoroughly before continuing

# there are a few ways to build signals to capture this
# there can be more than one working simultaneously to capture it
# pick a couple to capture dp dV then go build testing process

# - put stuff in neutral terms
# - do quicktests on signals
# - pursue accordingly
# mixture of z-scores and skews can capture a lot of this
# can test individually but when they combine it might get more interesting

# e.g. skew of returns to map overall profile & z-score of returns to understand how high current return is
# want to understand how extended is the move


# FINISH PREPARING PREDICTOR(S) VARIABLE(S)

# RETURNS (CHANGE IN PRICE)
#   ZSCORE (ALREADY NEUTRAL)
#   SKEW

# VOLUME (CHANGE IN VOLUME)
#   ZSCORE I.E. STANDARDISE (NEUTRAL / ZSCORE) - MAYBE JUST ZSCORE IS ENOUGH
#   SKEW

# FINSIH PREPARING RESPONSE VARIABLE
# CLEAN RETURNS (REMOVE TOP / BOTTOM 1-2.5%)
# APPLY SIGMOID FUNCTION

# SPLIT INTO BUCKETS LOOK AT NEUTRAL RET PROFILE

# BACKTEST

