import pandas as pd
import time
from zipline.data import bundles
from zipline.data.data_portal import DataPortal
from zipline.pipeline.data import USEquityPricing
from zipline.pipeline.engine import SimplePipelineEngine
from zipline.pipeline.loaders import USEquityPricingLoader
from zipline.utils.calendars import get_calendar
from zipline.assets._assets import Equity


# Set-Up Pricing Data Access
trading_calendar = get_calendar('NYSE')
bundle = 'quandl'
bundle_data = bundles.load(bundle)


loaders = {}

def my_dispatcher(column):
    return loaders[column]

pipeline_loader = USEquityPricingLoader(
    bundle_data.equity_daily_bar_reader,
    bundle_data.adjustment_reader,
)

def choose_loader(column):
    if column in USEquityPricing.columns:
        return pipeline_loader
    return my_dispatcher(column)

# Set-Up Pipeline Engine
engine = SimplePipelineEngine(
    get_loader=choose_loader,
    calendar=trading_calendar.all_sessions,
    asset_finder=bundle_data.asset_finder,
)

def run_pipeline(pipeline, start_date, end_date):
    return engine.run_pipeline(
        pipeline,
        pd.Timestamp(start_date, tz='utc'),
        pd.Timestamp(end_date, tz='utc')
    )

data = DataPortal(
    bundle_data.asset_finder,
    trading_calendar=trading_calendar,
    first_trading_day=bundle_data.equity_daily_bar_reader.first_trading_day,
    equity_minute_reader=None,
    equity_daily_reader=bundle_data.equity_daily_bar_reader,
    adjustment_reader=bundle_data.adjustment_reader,
)

def get_symbols(tickers, as_of_date=None):
    #TODO: this doesn't work for a list of tickers
    if (type(tickers)==str):
        return bundle_data.asset_finder.lookup_symbols(
            [tickers], as_of_date=as_of_date)
    else:
        if(type(tickers[0]==Equity)):
           return tickers
        else:
           return bundle_data.asset_finder.lookup_symbols(
               tickers, as_of_date=as_of_date)

def get_pricing(tickers, start_date, end_date, field='close'):

    end_dt = pd.Timestamp(end_date, tz='UTC', offset='C')
    start_dt = pd.Timestamp(start_date, tz='UTC', offset='C')
        
    symbols = get_symbols(tickers, as_of_date=end_dt)
    
    end_loc = trading_calendar.closes.index.get_loc(end_dt)
    start_loc = trading_calendar.closes.index.get_loc(start_dt)    
    
    dat = data.get_history_window(
        assets=symbols,
        end_dt=end_dt,
        bar_count=end_loc - start_loc,
        frequency='1d',
        field=field,
        data_frequency='daily'
    )

    return dat

import alphalens as al

def make_quantile_plot(df, start_date, end_date):
    assets = df.index.levels[1].values.tolist()
    df = df.dropna()
    pricing = get_pricing(
        assets,
        start_date,
        end_date,
        'close'
    )
    
    factor_names = df.columns
    factor_data = {}

    start_time = time.clock()
    for factor in factor_names:
        print("Formatting factor data for: " + factor)
        factor_data[factor] = al.utils.get_clean_factor_and_forward_returns(
            factor=df[factor],
            prices=pricing,
            periods=[1]
        )
    end_time = time.clock()
    print("Time to get arrange factor data: %.2f secs" % (end_time - start_time))
    
    qr_factor_returns = []

    for i, factor in enumerate(factor_names):
        mean_ret, _ = al.performance.mean_return_by_quantile(factor_data[factor])
        mean_ret.columns = [factor]
        qr_factor_returns.append(mean_ret)

    df_qr_factor_returns = pd.concat(qr_factor_returns, axis=1)

    (10000*df_qr_factor_returns).plot.bar(
        subplots=True,
        sharey=True,
        layout=(4,2),
        figsize=(14, 14),
        legend=False,
        title='Alphas Comparison: Basis Points Per Day per Quantile'
    )

    return df_qr_factor_returns
    

def make_factor_plot(df, start_date, end_date):
    assets = df.index.levels[1].values.tolist()
    df = df.dropna()
    pricing = get_pricing(
        assets,
        start_date,
        end_date,
        'close'
    )
    
    factor_names = df.columns
    factor_data = {}

    start_time = time.clock()
    for factor in factor_names:
        print("Formatting factor data for: " + factor)
        factor_data[factor] = al.utils.get_clean_factor_and_forward_returns(
            factor=df[factor],
            prices=pricing,
            periods=[1]
        )
    end_time = time.clock()
    print("Time to get arrange factor data: %.2f secs" % (end_time - start_time))
    
    ls_factor_returns = []

    start_time = time.clock()
    for i, factor in enumerate(factor_names):
        ls = al.performance.factor_returns(factor_data[factor])
        ls.columns = [factor]
        ls_factor_returns.append(ls)
    end_time = time.clock()
    print("Time to generate long/short returns: %.2f secs" % (end_time - start_time))

    df_ls_factor_returns = pd.concat(ls_factor_returns, axis=1)
    (1+df_ls_factor_returns).cumprod().plot(title='Cumulative Factor Returns');
    return df_ls_factor_returns

