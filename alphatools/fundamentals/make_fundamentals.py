from zipline.data import bundles
from zipline.pipeline import Pipeline
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.data import Column  
from zipline.pipeline.data import DataSet
from zipline.pipeline.engine import SimplePipelineEngine
from zipline.pipeline.filters import StaticAssets
from zipline.pipeline.loaders import USEquityPricingLoader
from zipline.pipeline.loaders.frame import DataFrameLoader
from zipline.utils.calendars import get_calendar

import numpy as np
import pandas as pd

trading_calendar = get_calendar('NYSE')
bundle_data = bundles.load('quandl')

data_file = 'path/to/sharadar/data'

df = pd.read_csv(data_file)#, nrows=1000)
df['Date'] = pd.to_datetime(df['Date'])

df['sid'] = np.nan
df = df.set_index('Date')

df.index = df.index.tz_localize('UTC')

dates = df.index.unique()

for day in dates:
    file_tickers = df.loc[day]['Ticker']
    sids = []
    for ticker in file_tickers:
        try:
            this_ticker = bundle_data.asset_finder.lookup_symbol(ticker, as_of_date=day)
            this_sid = this_ticker.sid
        except:
            this_sid = np.nan
        sids.append(this_sid)
    df.loc[day]['sid'] = sids

df.sid = df.sid.astype(float)
df = df.dropna()
df.sid = df.sid.astype(int)


df['MarketCap'] = pd.to_numeric(df['MarketCap'], errors='coerce')
df['P/B'] = pd.to_numeric(df['P/B'], errors='coerce')
df['P/S'] = pd.to_numeric(df['P/S'], errors='coerce')
df['P/E'] = pd.to_numeric(df['P/E'], errors='coerce')

# save the df
df.to_pickle('sharadar_with_sid.pkl')
