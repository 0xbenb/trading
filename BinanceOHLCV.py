from pycoingecko import CoinGeckoAPI
import pandas as pd
import direnv
import ccxt
import os
import time
import sqlalchemy
from sqlalchemy import exc
from sqlalchemy import create_engine

from CoreFunctions import *

# need to line up the id's from coingecko and first data sourc
direnv.load()

pd.options.display.max_rows = 10
pd.options.display.max_columns = 30
pd.options.display.width = 10000

def Instantiate_Exchange(exchange_id: str):
    """

    :param exchange_id: supported exchange id https://docs.ccxt.com/en/latest/exchange-markets.html
    :return:
    """
    exchange_class = getattr(ccxt, exchange_id)
    exchange = exchange_class({
        'apiKey': os.getenv(f'{exchange_id.upper()}_API_KEY'),
        'secret': os.getenv(f'{exchange_id.upper()}_SECRET_KEY'),
        'password': os.getenv(f'{exchange_id.upper()}_TRADING_PASSWORD'),
        'enableRateLimit': True
    })

    # globals()[name] = exchange

    return exchange


def Create_SQL_Engine():
    DB_USER = os.getenv('DB_USER')
    DB_PW = os.getenv('DB_PW')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT')
    DATABASE = os.getenv('DATABASE')

    engine = create_engine(f"""postgresql+psycopg2://{DB_USER}:{DB_PW}@{DB_HOST}:{DB_PORT}/{DATABASE}""")

    return engine


def Unpack_OHLCV(dat):
    if len(dat) == 0:
        pass
    else:
        dat = pd.DataFrame(dat, columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        dat['time'] = pd.to_datetime(dat['time'], unit='ms')

    return (dat)


def fetch_OHLCV(exchange_id, symbol_id, timeframe, date_min=None):
    exchange = Instantiate_Exchange(exchange_id)

    dat = exchange.fetch_ohlcv(symbol=symbol_id, timeframe='1w')
    dat = Unpack_OHLCV(dat)

    store = []

    start_t = int(dat['time'].min().value / 10 ** 6)
    ms_week = 604800000
    start_t = start_t - ms_week

    if date_min is not None:
        start_t = date_min
    # print('setting start_t = date_min')

    go = True

    while go:
        try:
            dat = exchange.fetch_ohlcv(symbol=symbol_id, since=start_t, timeframe=timeframe)
            dat = Unpack_OHLCV(dat)
        # print(start_t, dat)

        except Exception:
            # print('sleep')
            time.sleep(60)
            dat = exchange.fetch_ohlcv(symbol='XDB/USDT', since=start_t, timeframe=timeframe)
            dat = Unpack_OHLCV(dat)

        # print(dat['time'].max())

        if len(dat) == 0:
            go = False

        if start_t == int(dat['time'].max().value / 10 ** 6):
            go = False

        start_t = int(dat['time'].max().value / 10 ** 6)
        store.append(dat)

    store = pd.concat(store)
    store = store.drop_duplicates('time')
    store['symbol'] = symbol_id
    store['timeframe'] = timeframe
    store = store[['time', 'symbol', 'timeframe', 'open', 'high', 'low', 'close', 'volume']]

    return store


if __name__ == 'main':
    exchange_id = 'binance'
    exchange = Instantiate_Exchange(exchange_id)

    symbol_ids =


    timeframe = '1h'
    table_name = 'kucoin_ohlcv'

    SQL_engine = Create_SQL_Engine()

    try:
        db_table_info = DB_Table_Info(table_name=table_name, engine=SQL_engine)
        date_min = int(db_table_info['end_t'][0].timestamp() * 1000) + 1
        ohlcv = fetch_OHLCV(exchange_id=exchange_id, symbol_id=symbol_id, timeframe=timeframe, date_min=date_min)

    except Exception as e:
        print(str(e))
        ohlcv = fetch_OHLCV(exchange_id=exchange_id, symbol_id=symbol_id, timeframe=timeframe)

    SQL_engine = Create_SQL_Engine()

    if len(ohlcv) != 0:
        pop(data=ohlcv, table_name='kucoin_ohlcv', engine=SQL_engine)

# python -c "from Run_Programs import Run_Fetch_OHLCV; Run_Fetch_OHLCV()





































