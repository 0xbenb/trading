import pandas as pd
import direnv
import ccxt
import os
import time
import random
from datetime import datetime, timedelta
from CoreFunctions import *
direnv.load()
pd.options.display.max_rows = 10
pd.options.display.max_columns = 30
pd.options.display.width = 10000

engine = Create_SQL_Engine()
conn = Create_SQL_Connection(db_engine=engine)

# 7 DAY universe stationary definition of universe - will improve later

univ = Universe_Definition(top100_ndays_ago=7)