# This is a simple pair trade example. Let's
# look at the 10-year vs 5-year future.



import numpy as np
from sklearn.linear_model import LinearRegression
from zipline.api import (
    schedule_function,
    date_rules,
    time_rules,
    order_target_percent,
    record,
    symbol
)
    

LOOKBACK_DAYS = 120
ENTRY_THRESHOLD = 2.0
EXIT_THRESHOLD = 0.50

def initialize(context):
# This function runs once when sim starts.
# Put your `schedule_function()`, `set_slippage`, and `set_commission` calls here.

    # FV is the five-year root symbol
    # TU is the ten-year root symbol
    context.asset_A = symbol('AAPL')
    context.asset_B = symbol('MSFT')

    # Keep state if we have crossed the ENTRY_THRESHOLD
    context.position_initiated = False
    
    # Run rebal function every day 15min before
    # close.
    schedule_function(
        func=rebal,
        date_rule=date_rules.every_day(),
        time_rule=time_rules.market_close(
            minutes=15)
    )

def before_trading_start(context, data):
    # This function runs before each daily session
    pass

def rebal(context, data):
# Basic pair trade; price-based regression
 
    # get historical data for both futures
    hist = data.history(
        [context.asset_A, context.asset_B],
        fields='price',
        bar_count=LOOKBACK_DAYS,
        frequency='1d'
    )
    
    # the reshape is a scikit-learn nuanace when you have 1-dim data
    asset_A_prices = hist[context.asset_A].values.reshape(-1,1)
    asset_B_prices = hist[context.asset_B].values.reshape(-1,1)
    
    # run a price regression
    lm = LinearRegression().fit(
        asset_A_prices, # X
        asset_B_prices  # y
    )
    
    # get residuals
    residuals = asset_B_prices - lm.predict(asset_A_prices)
    
    # the most recent residual is the current spread
    current_spread = residuals[-1]
    
    # Z-Score of current spread
    score = (current_spread/np.nanstd(residuals))[-1]
    
    target_weights = {}

    # trading logic
    if score > ENTRY_THRESHOLD and not context.position_initiated:
        target_weights[context.asset_A] = -5.0  # i.e, 500%
        target_weights[context.asset_B] = 5.0
        context.position_initiated = True
    elif score < -ENTRY_THRESHOLD and not context.position_initiated:
        target_weights[context.asset_A] = 5.0
        target_weights[context.asset_B] = -5.0
        context.position_initiated = True
    elif np.abs(score) < EXIT_THRESHOLD and context.position_initiated:
        #unwind
        for futures_contract, position in context.portfolio.positions.items():
            target_weights[futures_contract] = 0
        context.position_initiated = False
    
    for futures_contract, target in target_weights.items():
        order_target_percent(futures_contract, target)

    record(A=asset_A_prices[-1][0])
    record(B=asset_B_prices[-1][0])
    record(score=score)

