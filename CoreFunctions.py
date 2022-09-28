import time
import pandas as pd
import direnv
import ccxt
import os
import yaml
from datetime import datetime
import sqlalchemy
from sqlalchemy import exc
from sqlalchemy import create_engine
import paramiko
import io
import numpy as np
import itertools
from pypika import Table, Query, Field, Order, functions as fn


# direnv.load()
#
# pd.options.display.max_rows = 10
# pd.options.display.max_columns = 30
# pd.options.display.width = 10000

def Create_SQL_Engine():
    DB_USER = os.getenv('DB_USER')
    DB_PW = os.getenv('DB_PW')
    DB_HOST = os.getenv('DB_HOST')
    DB_PORT = os.getenv('DB_PORT')
    DATABASE = os.getenv('DATABASE')

    db_engine = create_engine(f"""postgresql+psycopg2://{DB_USER}:{DB_PW}@{DB_HOST}:{DB_PORT}/{DATABASE}""")

    return db_engine

def Create_SQL_Connection(db_engine):
    try:
        db_conn = db_engine.raw_connection()

    except Exception as e:
        print(e)

    return db_conn

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


def Create_Database_Table(table_name: str, db_engine, db_conn):
    """

    :param table_name: which table is being inserted into
    :param db_engine: SQL engine
    :param db_conn: SQL connection
    :return: no return (check table)
    """
    db_conn = db_engine.raw_connection()
    cur = db_conn.cursor()

    q = f"""
    SELECT EXISTS (
    SELECT FROM information_schema.tables 
    WHERE  table_schema = 'public'
    AND    table_name   = '{table_name}'
    );"""

    exists = pd.read_sql_query(q, con=db_engine)

    if not exists['exists'].values[0]:

        if table_name == 'universe':
            q = """
            create table universe(
            time timestamptz NOT NULL, 
            coingecko_id text NOT NULL, 
            coingecko_symbol text NOT NULL, 
            coingecko_name text NOT NULL,
            binance_id text, 
            binance_symbol text, 
            binance_base text, 
            binance_quote text
            )
            """
            cur.execute(q)
            # cur.execute('rollback')
            db_conn.commit()

            q = f"""SELECT create_hypertable('{table_name}','time');"""
            cur.execute(q)
            db_conn.commit()

            # q = f"""CREATE UNIQUE INDEX {table_name}_index on {table_name}(time,uid);"""
            # cur.execute(q)
            # db_conn.commit()

        if table_name == 'binance_ohlcv':
            q = """
            create table binance_ohlcv(
            time timestamptz NOT NULL, 
            symbol text NOT NULL, 
            timeframe text NOT NULL, 
            o numeric, 
            h numeric, 
            l numeric, 
            c numeric, 
            volume numeric
            )
            """
            cur.execute(q)
            # cur.execute('rollback')
            db_conn.commit()

            q = f"""SELECT create_hypertable('{table_name}','time');"""
            cur.execute(q)
            db_conn.commit()

            q = f"""CREATE UNIQUE INDEX {table_name}_index on {table_name}(time,symbol,timeframe);"""
            cur.execute(q)
            db_conn.commit()


def pop(data: pd.DataFrame, table_name: str, db_engine):
    """

    :param data: data being inserted into database
    :param db_conn: SQL connection
    :param table_name: name of table being inserted into
    :return: no return (pop)
    """

    db_conn = Create_SQL_Connection(db_engine=db_engine)
    cur = db_conn.cursor()

    output = io.StringIO()
    data.to_csv(output, sep='\t', header=False, index=False)
    output.seek(0)
    contents = output.getvalue()
    cur.copy_from(output, table_name, null="")  # null values become ''
    db_conn.commit()

    db_conn.close() # might need to stop closing my connection if i want to keep it open for pops


def DB_Table_Info(table_name: str, db_engine):

    q = f"select min(time) start_t, max(time) end_t, symbol from {table_name} group by symbol"
    db_table = pd.read_sql_query(q, con=db_engine)

    return db_table


def Import_Info(table_name, db_engine, db_conn):
    # Get info from the existing table to know new data to pull

    q = f"select {table_name.split('_')[0]}_symbol as symbol from universe " \
        f"where time=(select max(time) from universe)"
    univ = pd.read_sql_query(q, con=db_engine).dropna()


    table_info = DB_Table_Info(table_name=table_name, db_engine=db_engine)

    table_info = univ.merge(table_info, how='left', on='symbol')

    # if current entry is most recent hourly remove from data pull
    current_ts = pd.Timestamp.utcnow()
    max_ts = pd.Timestamp(current_ts.strftime(format='%Y-%m-%d %H:00:00.000000'))

    import_info = table_info[table_info['end_t'] != str(max_ts)] # get rid of up to date fields

    return import_info


def DB_Query_Statement(table_name, columns=None, symbol=None, time_start=None, time_end=None):
    table = Table(table_name)

    # SELECT COLUMNS ELSE *
    if columns:
        columns = '","'.join(columns)
        q = Query.from_(table).select(columns)
    else:
        q = Query.from_(table_name).select('*')

    if symbol:
        q = q.where(table.uid.isin(symbol))
    if time_start:
        q = q.where(table.time >= time_start)
    if time_end:
        q = q.where(table.time <= time_end)

    # ORDER ACCORDINGLY
    if symbol:
        q = q.orderby('symbol', 'time')  # , order=Order.desc
    else:
        q = q.orderby('time')  # , order=Order.desc

    sql_statement = str(q)

    return sql_statement


def DB_Query(query: str, engine):
    conn = Create_SQL_Connection(engine)

    table_dat = pd.read_sql_query(query, con=conn)

    conn.close()

    return table_dat

