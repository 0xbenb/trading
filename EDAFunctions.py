import pandas as pd
import direnv
import ccxt
import os
import time
import random
from datetime import datetime, timedelta
from CoreFunctions import *
from scipy.stats import skew, zscore
import plotly.express as px
import plotly.graph_objects as go
from plotly.graph_objects import *
import plotly.io as pio
pio.renderers.default = "browser"
from plotly.subplots import make_subplots
from itertools import product

direnv.load()
pd.options.display.max_rows = 10
pd.options.display.max_columns = 30
pd.options.display.width = 10000

engine = Create_SQL_Engine()
conn = Create_SQL_Connection(db_engine=engine)

colour_theme = {'background': '#121212', 'tile': '#181818', 'top_gradient': '#404040',
                'bottom_gradient': '#282828', 'primary_text': '#FFFFFF', 'secondary_text': '#B3B3B3'}


def Sample_Data(data:pd.DataFrame, column: str, n_sample):
    subset = np.random.choice(data[column].unique(), n_sample)
    data = data[data[column].isin(subset)]

    return data


def Number_Observations_Time(data: pd.DataFrame, var_name):
    n_obs = data.groupby(['time'])[var_name].count().reset_index()

    config = dict({'scrollZoom': False, 'displaylogo': False, 'displayModeBar': False})

    # https://blog.karenying.com/posts/50-shades-of-dark-mode-gray

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=n_obs['time'], y=n_obs[var_name], fill='tozeroy'))  # override default markers+lines

    fig.update_layout({
        'paper_bgcolor': colour_theme['background'],
        'plot_bgcolor': colour_theme['top_gradient'],
        'font_family': "Courier New",
        'font_color': colour_theme['primary_text'],
        'title_text': f'<b>Observations through time: {var_name}',
        'titlefont': dict(size=30)
         })

    fig.update_xaxes(tickfont=dict(size=22), showgrid=False, title='time', zeroline=False, tickprefix='')
    fig.update_yaxes(tickfont=dict(size=22), showgrid=False, title='n observations', zeroline=False, tickprefix='')

    fig.update_yaxes(title_font=dict(size=30))
    fig.update_xaxes(title_font=dict(size=30)) # only way to get it update the size for now figure out why

    fig.show(config=config)

def Readable_Time_Window(t_window: int):
    # will assume t_window only hours for now and only convert to days if necessary

    if (t_window >= 24) & (t_window % 24 == 0):
        window_days = int(t_window / 24)
        readable_window = f'{window_days}d'

    else:
        readable_window = f'{t_window}h'

    return readable_window

# in terms of treating outliers, don't want to over clean / remove.
# generally raw data > calculate > standardise > normalise
# can ultimately test approach to see which option gives the best predictions

def Calculate_Skew(data: pd.DataFrame, variables: list, t_windows: list, bias: bool, min_obs: float):

    # pandas calculates unbiased skew i.e. bias = False (scipy defaults to true)
    # If bias = False, the calculations are corrected for bias
    # To make scipy.stats.skew compute the same value as the skew() method in Pandas, add the argument bias=False.
    # scipy calculates biased skew as default i.e. bias = True

    new_features = []
    for variable, t_window in zip(variables, t_windows):

        readable_t_window = Readable_Time_Window(t_window)

        if not bias:

            new_feature = f'{variable}_skew_{readable_t_window}'
            data[new_feature] = data.groupby('coin')[variable].\
                rolling(window=t_window, min_periods=round(min_obs*t_window)).\
                skew(skipna=True).reset_index(level=0, drop=True)

            new_features.append(new_feature)

        # for biased data
        # if bias:
        #     def calc_skew(x):
        #         return skew(x, nan_policy='omit', bias=False)
        #         # calculations are corrected for bias if bias=False (scipy defaults to True)
        #
        #     res = data.groupby('coin', group_keys=True)['ret_1h_neutral'].apply(
        #         lambda x: x.rolling(t_window).apply(calc_skew)
        #     )
        #     data['skew'] = res.reset_index(level=0, drop=True)

    return data, new_features

def Standardise(data: pd.DataFrame, method: str, variables: list, GroupBy: list, t_windows: list = None,
                min_obs: float = None):
    """

    :param data: data block of time, coin, variable, ...
    :param method: zscore
    :param t_windows: time window for calculation
    :param variables: variable to standardise
    :param GroupBy: variable(s) to groupby
    :param min_obs: number 0-1 for threshold of occurrences before calc
    :return:
    """

    if method == 'trailing_zscore':

        new_features = []
        for variable, t_window in zip(variables, t_windows):

            readable_t_window = Readable_Time_Window(t_window)
            new_feature = f'{variable}_zscore_{readable_t_window}'

            # (N-ddof) ddof = 0 for entire population (N) or ddof = 1 for sample (N-1) (n big variance sample = pop)
            data['mean'] = data.groupby(GroupBy)[variable].rolling(window=t_window, min_periods=round(min_obs*t_window)).\
                mean().reset_index(level=0, drop=True)
            data['std'] = data.groupby(GroupBy)[variable].rolling(window=t_window, min_periods=round(min_obs*t_window)).\
                std(ddof=0).reset_index(level=0, drop=True)
            data[new_feature] = (data[variable] - data['mean']) / data['std']

            data.drop(['mean', 'std'], axis=1, inplace=True)

            new_features.append(new_feature)

    if method == 'xzscore':

        new_features = []
        for variable in variables:

            new_feature = f'{variable}_xzscore'
            # (N-ddof) ddof = 0 for entire population (N) or ddof = 1 for sample (N-1) (n big variance sample = pop)
            data['mean'] = data.groupby(GroupBy)[variable].transform('mean')
            data['std'] = data.groupby(GroupBy)[variable].transform('std')
            data[new_feature] = (data[variable] - data['mean']) / data['std']

            data.drop(['mean', 'std'], axis=1, inplace=True)

            new_features.append(new_feature)

    return data, new_features


def Normalise(data: pd.DataFrame, variables: list, t_windows: int, method: str, min_obs: float):
    """

        :param data: data block of time, coin, variable, ...
        :param method: sigmoid, tanh, normal
        :param t_windows: time window for calculation
        :param variables: variable to standardise
        :return:
        """

    # https://stackoverflow.com/questions/51646475/how-to-normalize-training-data-for-different-activation-functions
    # sigmoid or tanh - some say tanh has better outlier treatment / down the line with models that like -ve's
    # e.g. neural nets with -ve hidden layers

    def norm(x):
        # normalise x to range [-1,1]
        x = np.array(x)
        nom = (x - x.min()) * 2.0
        denom = x.max() - x.min()
        return nom / denom - 1.0

    def sigmoid(x, k=0.1):
        # sigmoid function
        # use k to adjust the slope
        x = np.array(x)
        s = 1 / (1 + np.exp(-x / k))
        return s

    def tanh(x):
        x = np.array(x)
        m = np.nanmean(x, axis=0)
        std = np.nanstd(x, axis=0)

        t = 0.5 * (np.tanh(0.01 * ((x - m) / std)) + 1)

        return t

    new_features = []
    for variable, t_window in zip(variables, t_windows):

        if method == 'normal':
            new_feature = f'{variable}_norm'
            data[new_feature] = data.groupby('coin', group_keys=True)[variable].\
                rolling(window=t_window, min_periods=round(min_obs*t_window)).\
                apply(lambda x: norm(x)[-1]).reset_index(level=0, drop=True)

        if method == 'sigmoid':
            new_feature = f'{variable}_sigmoid'
            data[new_feature] = data.groupby('coin', group_keys=True)[variable].\
                rolling(window=t_window, min_periods=round(min_obs*t_window)).\
                apply(lambda x: sigmoid(x)[-1]).reset_index(level=0, drop=True)

        if method == 'tanh':
            new_feature = f'{variable}_tanh'
            data[new_feature] = data.groupby('coin', group_keys=True)[variable].\
                rolling(window=t_window, min_periods=round(min_obs*t_window)).\
                apply(lambda x: tanh(x)[-1]).reset_index(level=0, drop=True)

        new_features.append(new_feature)

    return data, new_features


def Remove_Outliers(data: pd.DataFrame, lower_upper_bounds: list, variables: list, GroupBy: list):
    """
    
    :param data: data block  
    :param GroupBy: list of items to groupby 
    :param lower_upper_bounds: list format e.g. [2.5, 97.5]
    :param variables: variable of column to remove outliers on
    :return: 
    """

    def outliers(s, replace=np.nan):
        # setting to 2.5% for now
        lower_bound, upper_bound = np.nanpercentile(s, lower_upper_bounds)

        return s.where((s > lower_bound) & (s < upper_bound), replace)

    new_features = []
    for variable in variables:
        new_feature = f'{variable}_rmoutliers'

        if GroupBy:
            data[new_feature] = data.groupby(GroupBy)[variable].apply(outliers)

        else:
            data[new_feature] = outliers(data[variable])

        new_features.append(new_feature)

    return data, new_features


def Create_Bins(data: pd.DataFrame, GroupBy: list, variables: list):
    """

    :param data: dataframe of data to create bins
    :param GroupBy: e.g. [time]
    :param variables: variable to create bins from
    :return:
    """

    # two scenarios cause problems
    # (1) all NA
    # (2) not enough (unique) observations per bucket
    # lots of ways to treat this i'm going to omit buckets without these

    new_features = []
    for variable in variables:
        new_feature = f'{variable}_bins'

        # will build this only for time grouping for now
        tmp = data.groupby(GroupBy).agg({variable: ['count', 'nunique']})
        tmp.columns = tmp.columns.map('_'.join)
        tmp.reset_index(inplace=True)
        tmp['rm_bins'] = np.where((tmp[f'{variable}_count'] < 5) | (tmp[f'{variable}_nunique'] < 5), 1, 0)
        rm_times = tmp.loc[tmp['rm_bins'] == 1]['time'].tolist()

        signal_bins = data.copy(deep=True)
        signal_bins = signal_bins[~signal_bins['time'].isin(rm_times)]
        signal_bins[new_feature] = signal_bins.groupby(GroupBy)[variable].transform(lambda x: pd.cut(x, bins=5, labels=range(1,6)))

        data = data.merge(signal_bins[['time', 'coin', new_feature]], how='left', on=['time', 'coin'])

        new_features.append(new_feature)

    return data, new_features


def Plot_Bins(data, bin_var, output_var):
    data = data[['time', 'coin', bin_var, output_var]].dropna()
    data[f'{output_var}_median'] = data.groupby('time')[output_var].transform('median')
    data[f'{output_var}_neutral'] = data[output_var] - data[f'{output_var}_median']

    bins_smy = data.groupby(bin_var)[f'{output_var}_neutral'].median().reset_index()

    fig = px.bar(bins_smy, x=bin_var, y=f'{output_var}_neutral')
    fig.show()

    return bins_smy


def Missing_Values_Plot(data):
    data = data.copy(deep=True)
    data[data.notnull()] = 0
    data[data.isnull()] = 1
    data[data == 0] = np.nan
    col_index = list(range(data.shape[1]))

    data = data + col_index

    fig = px.line(data, x=data.index, y=data.columns, title='Missing values')
    fig.update_layout(yaxis_title='Coin')
    fig.show()



def Mean_Variance_Plot(data: pd.DataFrame, t_window: int, min_obs: float, len_grid: int):
    data = data.copy(deep=True)

    if data.shape[1] > 10:
        sample_cols = list(data.columns)
        sample_cols = random.sample(sample_cols, len_grid*len_grid)
        data = data[sample_cols]

    r_mean = data.rolling(t_window, min_periods=round(min_obs * t_window)).mean()
    r_std = data.rolling(t_window, min_periods=round(min_obs * t_window)).std()

    fig = make_subplots(rows=len_grid, cols=len_grid, start_cell="bottom-left")

    grid_locs = list(range(1, len_grid+1)) # assuming equal grid for now
    grid_locs = list(product(grid_locs, grid_locs))

    for i in grid_locs:
        plot_number = grid_locs.index(i)
        r_mean_i = r_mean.iloc[:, plot_number]
        r_std_i = r_std.iloc[:, plot_number]

        fig.add_trace(go.Scatter(x=r_mean_i.index, y=r_mean_i, name=f'{r_mean_i.name}_mean'),
                      row=i[0], col=i[1])

        fig.add_trace(go.Scatter(x=r_std_i.index, y=r_std_i, name=f'{r_std_i.name}_std'),
                      row=i[0], col=i[1])

    fig.show()

