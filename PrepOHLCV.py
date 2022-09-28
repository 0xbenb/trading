import pandas as pd
import direnv
import ccxt
import os
import time
from CoreFunctions import *
direnv.load()
pd.options.display.max_rows = 10
pd.options.display.max_columns = 30
pd.options.display.width = 10000

engine = Create_SQL_Engine()
conn = Create_SQL_Connection(db_engine=engine)

q = DB_Query_Statement(table_name='universe', most_recent=True)

res = DB_Query(query=q, db_engine=engine)









































