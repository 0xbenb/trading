# Trading 

This is a top 100 Digital Asset trading strategy. Written in Python, it is a single exchange strategy (Binance). The code scales to most CEX exchanges (those supported by ccxt). It is deployed onto a Ubuntu 20.04.5 server. TimescaleDB extension is used for PostgreSQL database.

<br />

## Project Structure (To Date)

<pre>

├── UniverseCreation.py               // Aligns trading venues with coingecko top 100
├── CoreFunctions.py                  // Contains functions used across all programs 
├── <b>DataPrep</b>                   // Directory for data cleaning & preparation
│   └── PrepOHLCV.py                  // Preparing data from OHLCV endpoint 
│   └── PrepFunctions.py              // Contains functions used across Data Prep
├── <b>DataCollection</b>             // Directory for collecting data
│   └── BinanceOHLCV.py               // Collecting OHLCV data from Binance
├── <b>Deployment</b>                 // Directory for Deploying strategy
├── <b>EDA</b>                        // Directory for Exploratory Data Analysis
│   └── EDAFunctions.py               // Contains functions used across EDA
│   └── EDA.py                        // EDA process
├── <b>Model</b>                      // Directory for Model building
├── requirements.txt                  // Project requirements

</pre>

<br />

## Project Setup / Config 

To configure, create a .envrc file containing Database connection details and Exchange Keys laid out as so:

```
DB_USER=
DB_PW=
DB_HOST=
DB_PORT=
DATABASE=
BINANCE_API_KEY=
BINANCE_SECRET_KEY=<
```

## To Do 
> Implement logging <br />
> Fix universe creation to account for delisted pairs <br />

## Database Diagram (To Date)
![database diagram](https://user-images.githubusercontent.com/32384270/193476098-470a0efc-42e4-478b-b9dd-d3fb78b9a965.png) width=50% height=50

