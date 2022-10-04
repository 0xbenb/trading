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


def Calculate_Skew(data, variable, t_window, bias: bool):

    readable_t_window = Readable_Time_Window(t_window)
    # for unbiased data - pandas faster
    if not bias:
        data[f'{variable}_skew_{readable_t_window}'] = data.groupby('coin')[variable].\
            rolling(t_window).skew().reset_index(level=0, drop=True)

        return data

    # for biased data
    if bias:
        def calc_skew(x):
            return skew(x, bias=True)  # calculations are corrected for bias is bias=False (scipy defaults to True)

        res = data.groupby('coin', group_keys=True)['ret_1h_neutral'].apply(
            lambda x: x.rolling(t_window).apply(calc_skew)
        )
        data['skew'] = res.reset_index(level=0, drop=True)

        return data

def Standardise(data: pd.DataFrame, method: str, t_window: int, variable: str):
    """

    :param data: data block of time, coin, variable, ...
    :param method: zscore
    :param t_window: time window for calculation
    :param variable: variable to standardise
    :return:
    """

    readable_t_window = Readable_Time_Window(t_window)

    if method == 'zscore':

        # (N-ddof) ddof = 0 for entire population (N) or ddof = 1 for sample (N-1) (n big variance sample = pop)
        data['mean'] = data.groupby('coin')[variable].rolling(t_window).mean().reset_index(level=0, drop=True)
        data['std'] = data.groupby('coin')[variable].rolling(t_window).std(ddof=0).reset_index(level=0, drop=True)
        data[f'{variable}_zscore_{readable_t_window}'] = (data[variable] - data['mean']) / data['std']

        data.drop(['mean', 'std'], axis=1, inplace=True)

    return data


def Normalise(data: pd.DataFrame, method: str, t_window: int, variable: str):
    """

        :param data: data block of time, coin, variable, ...
        :param method: zscore
        :param t_window: time window for calculation
        :param variable: variable to standardise
        :return:
        """

    # sigmoid or tanh - some say tanh has more applications down the line with models that like -ve's
    # e.g. neural nets with -ve hidden layers

    # will use tried and trusted sigmoid for now [-1,1] range try tanh later
    def norm(x):
        # normalise x to range [-1,1]
        nom = (x - x.min()) * 2.0
        denom = x.max() - x.min()
        return nom / denom - 1.0

    def sigmoid(x, k=0.1):
        # sigmoid function
        # use k to adjust the slope
        s = 1 / (1 + np.exp(-x / k))
        return s

    unnormlised_data = data['ret_1h_neutral_skew_zscore']
    x = norm(unnormlised_data)










