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

    return dat


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

    # Exchange specifics
    exchange_id = 'binance'
    table_name = 'binance_ohlcv'
    timeframe = '1h'

    ## Move this content to function for finding what new data to import

    # Initialise clients
    exchange = Instantiate_Exchange(exchange_id)
    engine = Create_SQL_Engine()
    conn = Create_SQL_Connection(db_engine=engine)

    # Check the database table exists if not build
    Create_Database_Table(table_name='binance_ohlcv', db_engine=engine,
                          db_conn=conn)  # can do this at start of script

    import_info = Import_Info(table_name='binance_ohlcv', db_engine=engine, db_conn=conn)

    symbol_ids = import_info['symbol'].tolist()

    for symbol in symbol_ids:
        print(symbol)

        current_import = import_info[import_info['symbol'] == symbol]

        # if no data yet
        if pd.isnull(current_import['start_t'].iloc[0]):
            ohlcv = fetch_OHLCV(exchange_id=exchange_id, symbol_id=symbol, timeframe=timeframe)
            pop(data=ohlcv, table_name='binance_ohlcv', db_engine=engine)

        else:

            try:
                date_min = int(current_import.iloc[0,2].timestamp() * 1000) + 1
                ohlcv = fetch_OHLCV(exchange_id=exchange_id, symbol_id=symbol, timeframe=timeframe, date_min=date_min)

                if (len(ohlcv) != 0) and \
                        (str(ohlcv['time'].max()) != current_import.iloc[0]['end_t'].strftime("%Y-%m-%d %H:%M:%S")):
                    # some symbols stop getting supported should have a better way of filtering this out of universe
                    pop(data=ohlcv, table_name='binance_ohlcv', db_engine=engine)

            except Exception as e:
                print(str(e))

                # e == 'list indices must be integers or slices, not str':
                # this error is there is no new data e.g. USDC no longer supported by Binance
                # should have better process from removing from universe











































