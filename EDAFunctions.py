import pandas as pd
import direnv
import ccxt
import os
import time
import random
from datetime import datetime, timedelta
from CoreFunctions import *
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



