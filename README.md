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
├── CoreFunctions.py<br />
├── UniverseCreation.py<br />
├── **DataCleaning**<br />
├── **DataCollection**<br />
│   └── BinanceOHLCV.py<br />
├── **Deployment**<br />
├── **EDA**<br />
├── **Model**<br />
├── requirements.txt<br />


formatTreeString(
  {
    text: 'first',
    extra: 'extra',
    children: [
      {
        text: 'second',
        extra: 'another'
      },
      {
        text: 'third',
        children: [
          {
            text: 'fourth',
            extra: 'yet'
          },
          {
            text: 'fifth'
          },
          {
            text: 'sixth',
            extra: 'another',
            children: [
              {
                text: 'seventh',
                extra: 'one'
              },
              {
                text: 'eighth',
                extra: 'look',
                children: [
                  {
                    text: 'ninth',
                    extra: 'another'
                  },
                  {
                    text: 'tenth',
                    extra: 'one'
                  }
                ]
              }
            ]
          }
        ]
      },
      {
        text: 'eleventh',
        extra: 'yay'
      }
    ]
  },
  {
    guideFormat: chalk.dim
  }
);
