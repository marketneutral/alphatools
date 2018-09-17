import numpy as np
import pandas as pd

def L1_risk(wgts, returns):
    rets = returns.fillna(0.0).as_matrix()
    return np.mean(np.abs(rets.dot(wgts)))

def value_at_risk(
        weights,
        returns,
        alpha=0.95):
    
    returns = returns.fillna(0.0).as_matrix()
    portfolio_returns = returns.dot(weights)
    return np.percentile(portfolio_returns, 100 * (1-alpha))


def calc_portfolio_risk(context, data, risk_func, hist_days=180):
    positions = context.portfolio.positions
    positions_index = pd.Index(positions)
    share_counts = pd.Series(  
        index=positions_index,  
        data=[positions[asset].amount for asset in positions]  
    )

    current_prices = data.current(positions_index, 'price')  
    current_weights = (
        share_counts * current_prices / context.portfolio.portfolio_value
    )
    
    prices = data.history(
        current_weights.index.tolist(),
        'price',
        hist_days,
        '1d'
    )

    daily_rets = prices.pct_change()
    daily_rets = daily_rets - daily_rets.mean(skipna=True)
    daily_rets = daily_rets.fillna(0.0)

    risk = risk_func(current_weights.values, daily_rets)
    return risk
