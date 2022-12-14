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
from plotly.subplots import make_subplots
from plotly.graph_objects import *
import plotly.io as pio
pio.renderers.default = "browser"
from scipy.stats import skew
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from itertools import product
from statsmodels.tsa.stattools import adfuller


direnv.load()
pd.options.display.max_rows = 10
pd.options.display.max_columns = 30
pd.options.display.width = 10000

engine = Create_SQL_Engine()
conn = Create_SQL_Connection(db_engine=engine)

X = ['volume_1h_usd', 'ret_1h_neutral']
Y = ['fwd_ret_6h_neutral', 'fwd_ret_6h']

#########################
#  INTRO RESEARCH TOPIC #
#########################

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

skew_t_periods = [168, 168]
zscore_t_periods = [672, 672]

Number_Observations_Time(data=full_dat[['time', 'coin', X[1]]], var_name=X[1])

###############################
# PREPARE PREDICTOR VARIABLES #
###############################

full_dat = pd.read_csv('dat/full_dat.csv')  # instead of reimporting data from db

# CALCULATE SKEW
full_dat = Calculate_Skew(data=full_dat, variables=X, t_windows=skew_t_periods, bias=False, min_obs=0.5)
full_dat, skew_features = full_dat[0], full_dat[1]  # can wrap all processes into a fn / method later
# STANDARDISE
# trailing zscore
full_dat = Standardise(data=full_dat, method='trailing_zscore', variables=skew_features, t_windows=zscore_t_periods,
                       GroupBy=['coin'], min_obs=0.5)
full_dat, trailing_zscore_features = full_dat[0], full_dat[1]  # can wrap all processes into a fn / method later
# xscore
full_dat = Standardise(data=full_dat, method='xzscore', variables=skew_features,
                       GroupBy=['time'])
full_dat, xscore_features = full_dat[0], full_dat[1]  # can wrap all processes into a fn / method later

# NORMALISE
full_dat = Normalise(data=full_dat, variables=trailing_zscore_features, t_windows=zscore_t_periods,
                     method='tanh', min_obs=0.5)
full_dat, norm_features = full_dat[0], full_dat[1]  # can wrap all processes into a fn / method later
# REMOVE OUTLIERS
full_dat = Remove_Outliers(data=full_dat, lower_upper_bounds=[2.5, 97.5], variables=Y, GroupBy=['time'])
full_dat, rmoutlier_features = full_dat[0], full_dat[1]
# SIGNAL BINS
predictors = skew_features + trailing_zscore_features + xscore_features + norm_features
full_dat = Create_Bins(data=full_dat, GroupBy=['time'], variables=predictors)
full_dat, predictors_bins = full_dat[0], full_dat[1]

responses = Y + rmoutlier_features

################################
# STORE DATA TO SAVE RERUNNING #
################################

# full_dat.to_csv('dat/processed_dat.csv', index=False)
full_dat = pd.read_csv('dat/processed_dat.csv')

predictors = ['volume_1h_usd_skew_7d', 'ret_1h_neutral_skew_7d', 'volume_1h_usd_skew_7d_zscore_28d',
              'ret_1h_neutral_skew_7d_zscore_28d', 'volume_1h_usd_skew_7d_xzscore', 'ret_1h_neutral_skew_7d_xzscore',
              'volume_1h_usd_skew_7d_zscore_28d_tanh', 'ret_1h_neutral_skew_7d_zscore_28d_tanh']

responses = ['fwd_ret_6h_neutral', 'fwd_ret_6h', 'fwd_ret_6h_neutral_rmoutliers', 'fwd_ret_6h_rmoutliers']

###################################
# TEST PREDICTORS X VS RESPONSE Y #
###################################

pred_var = predictors[5]
resp_var = responses[1]

X = full_dat.loc[:, ['time', 'coin', pred_var]]
X = pd.pivot(data=X, index='time', columns='coin', values=pred_var)

Y = full_dat.loc[:, ['time', 'coin', resp_var]]
Y = pd.pivot(data=Y, index='time', columns='coin', values=resp_var)


# MISSING VALUES
Missing_Values_Plot(data=X)
# universe has better representation 2019-2020+


# DISTRIBUTION OF PREDICTOR
px.histogram(X, nbins=15)


# VISUALISE STATIONARITY
Mean_Variance_Plot(data=X, t_window=90*24, min_obs=0.5, len_grid=2)


# TEST STATIONARITY
res = Test_Stationarity(data=X, variable_name=pred_var, sample_size=5)

# adfuller you want output to be TRUE i.e. CAN't reject NULL => stationary
# kpss you want output to be FALSE i.e. CAN reject NULL => stationary
# it makes sense variables are stationary as they are built from neutral variables i.e. stripping out market movement
# or standardising / normalising approaches applied


# UNDERSTAND AUTOCORRELATION
Autocorrelation_Plot(X['DOGE'])
Autocorrelation_Plot(X['DOGE'], plot_pacf=True)
# https://www.kaggle.com/code/iamleonie/time-series-interpreting-acf-and-pacf
# shows high degree of autocorrelation
# i suppose one thing this means is good stability of the signal
# need to factor in this autocorrelation with model building e.g. autoregression


# TRANSITION MATRIX PREDICTOR
pred_bin = f'{pred_var}_bins'
sequences = Prepare_State_Sequences(data=full_dat, pred_bin=pred_bin)

tm = Calculate_Transition_Matrix(sequences)
tm = tm.iloc[1:, 1:]

px.imshow(tm, color_continuous_scale='RdBu_r')
# this looks logical and shows that from state to state it is more likely that the next state will be the adjacent state
# along a continuous scale. looks good, essentially confirming the autocorrelation in another way

# can develop this further by looking at longer time intervals


# EXPECTED VALUE RESPONSE Y TO PREDICTOR X
Plot_Bins(data=full_dat, bin_var=pred_bin, output_var=resp_var)
full_dat[['time', 'coin', pred_bin]][pred_bin].value_counts()

# TRANSITION MATRIX PREDICTOR VS RESPONSE
# previous code was setup with a shift of 1 but for this i'm actually going to create the current (i) to next(j)
# but next (j) is actually going to be the fwd return over some time period discretised
# will need to tweak the code for that to build state_current --> state_next

# Loose ends / Reminders
# factor in 1h constraint for putting on positions
# check stability of bins calc

# i think it makes sense that the variables i've tested so far are stationary as they are built from neutral rets

# random thoughts
# correlation of predictors vs output
# co correlation of predictors
# covariance of predictors

# Docs & Reference
# https://medium.com/@dhirajreddy13/stock-price-prediction-and-forecast-using-lstm-and-arima-52db753a23c7
# https://www.business-science.io/code-tools/2021/07/19/modeltime-panel-data.html
# https://www.kaggle.com/code/nholloway/stationarity-smoothing-and-seasonality/notebook
# https://www.kaggle.com/code/andreshg/timeseries-analysis-a-complete-guide
