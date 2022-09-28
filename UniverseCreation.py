from pycoingecko import CoinGeckoAPI
import pandas as pd
import direnv
import ccxt
import os
import io
import sqlalchemy
from sqlalchemy import exc
from sqlalchemy import create_engine
from datetime import datetime, timedelta
from CoreFunctions import *

direnv.load()

pd.options.display.max_rows = 10
pd.options.display.max_columns = 30
pd.options.display.width = 10000

def Exchange_Supported_Symbols(exchange_id: str):
    """

    :param exchange_id: string for exchange name
    :return:
    """
    # start with binance
    exchange = Instantiate_Exchange(exchange_id)  # only binance for now
    markets = exchange.load_markets()
    tuples = list(ccxt.Exchange.keysort(markets).items())

    supported_symbols = []
    for (k, v) in tuples:
        row = [v['id'], v['symbol'], v['base'], v['quote']]
        supported_symbols.append(row)
    supported_symbols = pd.DataFrame(supported_symbols, columns=['id', 'symbol', 'base', 'quote'])

    return supported_symbols


def Unsupported_Symbols(db_engine):

    # pick any binance table to filter down symbols no longer supported
    table_info = DB_Table_Info(table_name='binance_ohlcv', db_engine=db_engine)
    ts_7d_ago = str(datetime.utcnow() - timedelta(days=7))

    symbols = table_info[table_info['end_t'] < ts_7d_ago]
    symbols = symbols['symbol'].tolist()

    return symbols


# for the time being only using binance so don't need this code to scale to multiple
def Align_Universes():
    cg = CoinGeckoAPI()
    cg_ids = pd.DataFrame(cg.get_coins(per_page=100))
    cg_ids = cg_ids[['id', 'symbol', 'name']]
    cg_ids = cg_ids.add_prefix('coingecko_')

    supported_symbols = Exchange_Supported_Symbols(exchange_id='binance')

    quote_ccy = ['BTC', 'ETH', 'USDT'] # only interested in these quotes from binance as delisting usdc

    supported_symbols = supported_symbols[supported_symbols['quote'].isin(quote_ccy)]

    base_ccy = cg_ids['coingecko_symbol'].str.upper().tolist()
    base_ccy = [i for i in base_ccy if i not in quote_ccy]

    supported_symbols = pd.concat(
        [supported_symbols[supported_symbols['symbol'].isin(['BTC/USDT', 'ETH/USDT', 'ETH/BTC'])],
         supported_symbols[supported_symbols['base'].isin(base_ccy)]]).reset_index(drop=True)

    supported_symbols = supported_symbols.add_prefix('binance_')

    aligned_univ = pd.merge(cg_ids, supported_symbols, how='left', left_on=cg_ids['coingecko_symbol'],
                                 right_on=supported_symbols['binance_base'].str.lower())
    aligned_univ.drop('key_0', axis=1, inplace=True)
    # base_ccy_missing = [i for i in base_ccy if i not in supported_symbols['base'].tolist()]

    aligned_univ['time'] = pd.Timestamp.utcnow()

    col = aligned_univ.pop("time")
    aligned_univ.insert(0, col.name, col)

    return aligned_univ


if __name__ == "__main__":

    engine = Create_SQL_Engine()
    conn = Create_SQL_Connection(engine)

    univ = Align_Universes()
    unsupported_symbols = Unsupported_Symbols(db_engine=engine)

    univ = univ[~univ['binance_symbol'].isin(unsupported_symbols)]

    Create_Database_Table(table_name='universe', db_engine=engine, db_conn=conn)

    pop(data=univ, table_name='universe', db_engine=engine)