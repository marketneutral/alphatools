import numpy as np
import pandas as pd

def L1_risk(wgts, returns):
    """
    Returns the "L1" risk as per Konno & Yamazaki; aka the mean-absolute-
    deviation.
    """
    rets = returns.fillna(0.0).as_matrix()
    return np.mean(np.abs(rets.dot(wgts)))

def value_at_risk(
        weights,
        returns,
        alpha=0.95):
    """
    Returns the historical simulation VaR at the confidence threshold.
    """
    returns = returns.fillna(0.0).as_matrix()
    portfolio_returns = returns.dot(weights)
    return np.percentile(portfolio_returns, 100 * (1-alpha))

def expected_shortfall(
        weights,
        returns,
        alpha=0.95):
    """
    Returns the historical simulation CVaR at the confidence threshold.
    """

    var = value_at_risk(weights, returns, alpha)
    returns = returns.fillna(0.0)
    portfolio_returns = returns.dot(weights)
    return np.nanmean(portfolio_returns[portfolio_returns < var])


def calc_portfolio_risk(
        context,
        data,
        risk_func,
        hist_days=180,
        **kwargs):
    """
    This is a helper function designed to be the primary call in an algo
    for calculating portfolio-level risk. It takes the current context and
    data objects and a `risk_func` (e.g., `value_at_risk`), and formats 
    portfolio weights and returns such that the indicies line up in a numpy
    array.
    """

    
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

    risk = risk_func(current_weights.values, daily_rets, **kwargs)
    return risk
