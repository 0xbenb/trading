# trading

This python project is a top 100 trading strategy on binance. It is deployed onto a Ubuntu 20.04.5 server. TimescaleDB extension is used for PostgreSQL database.

<br />

## Project Setup / Config 

To configure, create a .envrc file containing Database connection details and Exchange Keys laid out as so:

DB_USER=<br />
DB_PW=<br />
DB_HOST=<br />
DB_PORT=<br />
DATABASE=<br />
BINANCE_API_KEY=<br /> 
BINANCE_SECRET_KEY=<br />

<br />

## Project Structure (working progress)
```
├── CoreFunctions.py          // Contains functions used across scripts
├── UniverseCreation.py       // Aligns trading venues with coingecko top 100 
├── DataCleaning              // Directory containing scripts for cleaning data
├── DataCollection            // Directory containing scripts for collecting data
│   └── BinanceOHLCV.py       // Collecting OHLCV data from Binance 
├── Deployment                // Director for Deploying strategy
├── EDA                       // Directory containing Exploratory Data Analysis 
├── Model                     // Directory for building models 
├── requirements.txt          // Project requirements 
```
