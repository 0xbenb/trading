from pycoingecko import CoinGeckoAPI
import pandas as pd
import direnv
import ccxt
import os
import time

from CoreFunctions import *

# need to line up the id's from coingecko and first data source
direnv.load()

pd.options.display.max_rows = 10
pd.options.display.max_columns = 30
pd.options.display.width = 10000


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
            dat = exchange.fetch_ohlcv(symbol=symbol_id, since=start_t, timeframe=timeframe)
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


if __name__ == '__main__':
    exchange_id = 'binance'
    table_name = 'binance_ohlcv'
    timeframe = '1h'

    symbol_id = 'KSM/USDT'

    exchange = Instantiate_Exchange(exchange_id)

    engine = Create_SQL_Engine()
    conn = Create_SQL_Connection(db_engine=engine)

    q = "select * from universe_mapping where time=(select max(time) from universe_mapping)"
    univ = pd.read_sql_query(q, con=engine)

    table_info = DB_Table_Info(table_name=table_name, db_engine=engine)

    symbol_ids = univ['binance_symbol'].drop_duplicates().dropna().tolist()

    for s in symbol_ids:

        try:
            date_min = int(table_info[table_info['symbol'] == s]['end_t'][0].timestamp() * 1000) + 1
            ohlcv = fetch_OHLCV(exchange_id=exchange_id, symbol_id=symbol_id, timeframe=timeframe, date_min=date_min)

        except Exception as e:
            print(str(e))
            ohlcv = fetch_OHLCV(exchange_id=exchange_id, symbol_id=symbol_id, timeframe=timeframe)

        if len(ohlcv) != 0:
            Create_Database_Table(table_name='binance_ohlcv', db_engine=engine,
                                  db_conn=conn)  # can do this at start of script
            pop(data=ohlcv, table_name='binance_ohlcv', db_conn=conn)









































