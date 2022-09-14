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

        if table_name == 'universe_mapping':
            q = """
            create table universe_mapping(
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


def pop(data: pd.DataFrame, table_name: str, db_conn):
    """

    :param data: data being inserted into database
    :param db_conn: SQL connection
    :param table_name: name of table being inserted into
    :return: no return (pop)
    """

    cur = db_conn.cursor()

    output = io.StringIO()
    data.to_csv(output, sep='\t', header=False, index=False)
    output.seek(0)
    contents = output.getvalue()
    cur.copy_from(output, table_name, null="")  # null values become ''
    db_conn.commit()

    db_conn.close()





