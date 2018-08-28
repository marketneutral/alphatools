import pandas as pd
from research_tools.research_tools import loaders 
from zipline.pipeline.data import Column
from zipline.pipeline.data import DataSet
from zipline.pipeline.loaders.frame import DataFrameLoader

from os import path
myfile_path = path.join(path.dirname(__file__), 'myfile.txt')

df = pd.read_pickle(path.join(path.dirname(__file__), 'sharadar_with_sid.pkl'))

MarketCap_frame = (
    df[['MarketCap', 'sid']].
    reset_index().set_index(['Date', 'sid']).
    unstack()
)

MarketCap_frame.columns = MarketCap_frame.columns.droplevel()

PriceToBook_frame = df[['P/B', 'sid']].reset_index().set_index(['Date', 'sid']).unstack()
PriceToBook_frame.columns = PriceToBook_frame.columns.droplevel()

PriceToSales_frame = df[['P/S', 'sid']].reset_index().set_index(['Date', 'sid']).unstack()
PriceToSales_frame.columns = PriceToSales_frame.columns.droplevel()

PriceToEarnings_frame = df[['P/E', 'sid']].reset_index().set_index(['Date', 'sid']).unstack()
PriceToEarnings_frame.columns = PriceToEarnings_frame.columns.droplevel()

class Fundamentals(DataSet):
    MarketCap = Column(dtype=float)
    PriceToBook = Column(dtype=float)
    PriceToSales = Column(dtype=float)
    PriceToEarnings = Column(dtype=float)

# register the loaders
loaders[Fundamentals.MarketCap] = DataFrameLoader(Fundamentals.MarketCap, MarketCap_frame)
loaders[Fundamentals.PriceToBook] = DataFrameLoader(Fundamentals.PriceToBook, PriceToBook_frame)
loaders[Fundamentals.PriceToSales] = DataFrameLoader(Fundamentals.PriceToSales, PriceToSales_frame)
loaders[Fundamentals.PriceToEarnings] = DataFrameLoader(Fundamentals.PriceToEarnings, PriceToEarnings_frame)
