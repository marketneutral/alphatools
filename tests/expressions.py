from alphatools.research import run_pipeline, make_factor_plot
from alphatools.expression import ExpressionAlpha
from alphatools.ics import Sector, SubIndustry
from zipline.pipeline.factors import AverageDollarVolume, CustomFactor, Returns
from zipline.pipeline.data import USEquityPricing as USEP
from zipline.pipeline import Pipeline

import pandas as pd
import numpy as np

universe = AverageDollarVolume(window_length=120).top(10)

expressions = {
    '0': 'close',
    '1': 'delay(close,1)',
    '2': 'delta(close,5)',
    '3': 'returns',
    '4': 'delta(close,1)/delay(close,1)',
    '5': 'delta(close,5)/delay(close,5)',
    '6': 'rank(close)',
    '7': 'indneutralize(close, IndClass.sector)',
}

class Control_1(CustomFactor):
    window_length=2
    inputs=[USEP.close]

    def compute(self, today, assets, out, close):
        out[:]=close[-2]

class Control_2(CustomFactor):
    window_length=6
    inputs=[USEP.close]

    def compute(self, today, assets, out, close):
        out[:]=close[-1] - close[-6]

control = {}
control_0 = USEP.close.latest
control_1 = Control_1()
control_2 = Control_2()
control_3 = Returns(window_length=2)
control_4 = Returns(window_length=2)
control_5 = Returns(window_length=6)
control_6 = USEP.close.latest.rank(mask=universe)
control_7 = USEP.close.latest.demean(groupby=Sector(), mask=universe)

control = {
    '0': control_0,
    '1': control_1,
    '2': control_2,
    '3': control_3,
    '4': control_4,
    '5': control_5,
    '6': control_6,
    '7': control_7,
}


start_date = '2017-01-04'
end_date = '2017-01-04'


def test_factor(expression, control, start_date='2017-01-04', end_date='2017-01-04', show_df=False):
    p = Pipeline(screen=universe)
    p.add(expression.make_pipeline_factor().pipeline_factor(mask=universe), 'expression_alpha')
    p.add(control, 'pipeline_factor')
    df = run_pipeline(p, start_date, end_date)
    print(np.allclose(df['expression_alpha'].values, df['pipeline_factor'].values))
    if show_df:
        print df


start_fac = 7
end_fac = 7

for i in range(start_fac, end_fac+1):
    test_factor(ExpressionAlpha(expressions[str(i)]), control[str(i)], show_df=True)
