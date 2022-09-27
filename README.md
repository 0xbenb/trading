<<<<<<< HEAD
# Trading 

This is a top 100 Digital Asset trading strategy. Written in Python, it is a single exchange strategy (Binance). The code scales to most CEX exchanges (those supported by ccxt). It is deployed onto a Ubuntu 20.04.5 server. TimescaleDB extension is used for PostgreSQL database.
=======
# Trading

This is a top 100 Digital Asset trading strategy. Written in Python, it is a single exchange 
strategy (Binance). The code scales to most CEX exchanges (those supported by ccxt). It is deployed 
onto a Ubuntu 20.04.5 server. TimescaleDB extension is used for PostgreSQL database.
>>>>>>> 877a409a5f3a8bbeab315144415f3e0cb650800a

<br />

## Project Structure (To Date)
<pre>
├── CoreFunctions.py                  // Contains functions used across scripts
<<<<<<< HEAD
├── UniverseCreation.py               // Aligns trading venues with coingecko top 100 
├── <b>DataCleaning</b>               // Directory containing scripts for cleaning data
├── <b>DataCollection</b>             // Directory containing scripts for collecting data
│   └── BinanceOHLCV.py               // Collecting OHLCV data from Binance 
├── <b>Deployment</b>                 // Director for Deploying strategy
├── <b>EDA</b>                        // Directory containing Exploratory Data Analysis 
├── <b>Model</b>                      // Directory for building models 
├── requirements.txt                  // Project requirements 
=======
├── UniverseCreation.py               // Aligns trading venues with coingecko top 100
├── <b>DataCleaning</b>               // Directory containing scripts for cleaning data
├── <b>DataCollection</b>             // Directory containing scripts for collecting data
│   └── BinanceOHLCV.py               // Collecting OHLCV data from Binance
├── <b>Deployment</b>                 // Director for Deploying strategy
├── <b>EDA</b>                        // Directory containing Exploratory Data Analysis
├── <b>Model</b>                      // Directory for building models
├── requirements.txt                  // Project requirements
>>>>>>> 877a409a5f3a8bbeab315144415f3e0cb650800a
</pre>

<br />

<<<<<<< HEAD
## Project Setup / Config 

To configure, create a .envrc file containing Database connection details and Exchange Keys laid out as so:
=======
## Project Setup / Config

To configure, create a .envrc file containing Database connection details and Exchange Keys laid out 
as so:
>>>>>>> 877a409a5f3a8bbeab315144415f3e0cb650800a
```
DB_USER=
DB_PW=
DB_HOST=
DB_PORT=
DATABASE=
BINANCE_API_KEY=
BINANCE_SECRET_KEY=<
```
<<<<<<< HEAD




=======
>>>>>>> 877a409a5f3a8bbeab315144415f3e0cb650800a
