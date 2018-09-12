<img src="https://user-images.githubusercontent.com/16124573/45173356-f3b9f180-b1d5-11e8-97ba-5e92154c630a.png" width="400">

[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

This package provides functions to make the equity alpha factor research process more accessible and productive. Convenience functions sit on top of [zipline](https://github.com/quantopian/zipline) and, specifically, the [`Pipeline`](https://www.quantopian.com/help#pipeline-api) cross-sectional classes and functions in that package. `alphatools` allows you to 

- `run_pipeline` in a Jupyter notebook (or from any arbitrary Python code) **in your local environment**,
- create `Pipeline` factors **at runtime** on **arbitrary data sources** (just expose the endpoint for data sitting somewhere, specify the schema, and...it's available for use in `Pipeline`!), and 
- parse and compile **"expression" style alphas** as described the paper ["101 Formulaic Alphas"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2701346) into `Pipeline` factors.

For example, with `alphatools`, you can, say, within a Jupyter notebook,

```python
from alphatools.research import run_pipeline
from alphatools.ics import Sector
from alphatools.data import Factory
from alphatools.expression import ExpressionAlpha
from zipline.pipeline.data import USEquityPricing as USEP
from zipline.pipeline.factors import Returns, AverageDollarVolume
from zipline.pipeline import Pipeline

universe = AverageDollarVolume(window_length=120).top(500)

my_factor = (
    -Returns(mask=universe, window_length=5).
    demean(groupby=Sector()).
    rank()
)

expr_factor = (
    ExpressionAlpha(
        'rank(indneutralize(-log(close/delay(close, 4))),IndClass.sector)'
    ).pipeline_factor(mask=universe)
)

p = Pipeline(screen=universe)

p.add(my_factor, '5d_MR_Sector_Neutral_Rank')
p.add(expr_factor, '5d_MR_Expression Alpha')

p.add(Factory['my_special_data'].value.latest.zscore(), 'Special_Factor')

start_date = '2017-01-04'
end_date = '2017-12-28'

df = run_pipeline(p, start_date, end_date)
```

## Bring Your Own Data

To "Bring Your Own Data", you simply point the Factory object to an endpoint and specify the schema. This is done by adding to the `json` file `data_sources.json`. For example, if you have a `csv` file on disk, `data.csv`, and a PostgreSQL table somewhere else, you would create `data_sources.json` as

```json
{
	"my_special_data": {
		"url": "/full/path/to/data/data.csv",
		"schema": "var * {asof_date: datetime, sid: int64, value: float64}"
	},
	
	"my_database_data": {
		"url": "postgresql://$USER:$PASS@hostname::my-table-name",
		"schema": "var * {asof_date: datetime, sid: int64, price_to_book: float64}"
}
```

In the case of the example PostgreSQL `url`, note that the text `$USER` will be substituted with the text in the environment variable `USER` and the text `$PASS` will be substituted with the text in the environment variable `PASS`. Basically, any text token in the `url` which is preceeded by `$` will be substituted by the text in the environment variable of that name. Hence, you do not need to expose actual credentials in this file.

The `schema` is specified as a `dshape` from the package `datashape` (docs [here]()). The magic happens via the `blaze/datashape/odo` stack. You can specify the `url` to a huge variety of source formats including `json`, `csv`, PostgreSQL tables, MongoDB collections, `bcolz`, Microsoft Excel(!?), `.gz` compressed files, collections of files (e.g., `myfiles_*.csv`), and remote locations like Amazon S3 and a Hadoop Distributed File System. To me, the [`odo`](https://en.wikipedia.org/wiki/Odo_(Star_Trek)) [documentation on URI strings](http://odo.pydata.org/en/latest/uri.html) is the clearest explanation on this.

Note that this data must be mapped to the `sid` as mapped by `zipline ingest`. Also, the data rowwise dates must be in a column titled `asof_date`. You can then access this data like

```python
from alphatools.data import Factory
	:
	:
	:
	
my_factor = Factory['my_database_data'].price_to_book.latest.rank()
p.add(my_factor)
```

This functionality should allow you to use new data in research very quickly with the absolute minimal amount of data engineering and/or munging. For example, commercial risk model providers often provide a single file per day for factor loadings (e.g., `data_yyyymmdd_fac.csv`). After `sid` mapping and converting the date column name to `asof_date`,  this data can be immediately available in `Pipeline` by putting a `url` in `data_sources.json` like `"url": "/path/to/dir/data_*_fac.csv"`, and `schema` like `"var * {asof_date: datetime, sid: int64, MKT_BETA: float64, VALUE: float64, MOMENTUM: float64, ST_REVERSAL: float64 ..."`.

## Expression Alphas

The ability to parse "expression" alphas is meant to help speed the research process and/or allow financial professionals with minimal Python experience to test alpha ideas. See ["101 Formulaic Alphas"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2701346) for details on this DSL. The (EBNF) grammar is fully specified ["here"](https://github.com/marketneutral/alphatools/blob/master/alphatools/expression/expression.lark). We use the `Lark` Python [parsing library](https://github.com/lark-parser/lark) (great name, no relation). Currently, the data for `open`, `high`, `low`, `close`, `volume` are accessible; the following calculations and operators are implemented

* `vwap`: the daily vwap (as a default, this is approximated with `(close + (opens + high + low)/3)/2`).
* `returns`: daily close-to-close returns.
* `+`,`-`, `*`, `/`, `^`: as expected, though only for two terms (i.e., only \<expr\> \<op\> \<expr\>); `^` is exponentiation, not bitwise or.
* `-x`: unary minus on x (i.e., negation).
* `abs(x)`, `log(x)`, `sign(x)`: elementwise standard math operations.
* `>`, `<`, `==`, `||`: elementwise comparator operations returning 1 or 0.
* `x ? y : z`: C-style ternary operator; `if x: y; else z`.
* `rank(x)`: scaled ranks, per day, across all assets (i.e., the cross-sectional rank); ranks are descending such that the rank of the maximum raw value in the vector is 1.0; the smallest rank is 1/N. The re-scale of the ranks to the interval [1/N,1] is implied by Alpha 1: 0.50 is subtracted from the final ranked value. The ordinal method is used to match `Pipeline` method `.rank()`.
* `delay(x, days)`: *x* lagged by *days*. Note that the *days* parameter in `delay` and `delta` differs from the `window_length` parameter you may be familiar with in `Pipeline`. The `window_length` refers to a the number of data points in the (row axis of the) data matrix, *not* the number of days lag. For example, in `Pipeline` if you want daily returns, you specify a `window_length` of `2` since you need 2 data points--today and the day prior--to get a daily return. In an expression alpha, the *days* is the lag *from today*. Concretely, a simple example to show is: the `Pipeline` factor `Returns(window_length=2)` is precisely equal to the expression alpha `delta(close,1)/delay(close,1)`.
* `correlation(x, y, days)`: the Pearson correlation of the values for assets in *x* to the corresponding values for the same assets in *y* over *days*; note this is very slow in the current implementation.
* `covariance(x, y, days)`: the covariance of the values for assets in *x* to the corresponding values for the same assets in *y* over *days*; note this is very slow as well currently.
* `delta(x, days)`: diff on *x* per *days* timestep.
* `signedpower(x, a)`: elementwise `sign(x)*(abs(x)^a)`.
* `decay_linear(x, days)`: weighted sum of *x* over the past *days* with linearly decaying weights (weights sum to 1; max of the weights is on the most recent day).
* `indneutralize(x, g)`: `x`, cross-sectionally "neutralized" (i.e., demeaned) against the group membership classifier `g`. `g` must be in the set {`IndClass.sector`, `IndClass.industry`, `IndClass.subindustry`}. The set `g` maps to the `Pipeline` classifiers `Sector()` and `SubIndustry()` in `alphatools.ics`. Concretely, the `Pipeline` factor `Returns().demean(groupby=Sector())` is equivalent (save a corner case on NaN treatment) to the expression `indneutralize(returns, IndClass.sector)`. If you do not specifically pass a token for `g`, the default of `IndClass.industry` is applied.
* `ts_max(x, days)`: the per asset time series max on *x* over the trailing *days* (also `ts_min(...)`).
* `max(a, b)`: The paper says that `max` is an alias for `ts_max(a, b)`; I think this is an error. Alphas 71, 73, 76, 87, and 96 do not parse with `max` as alias for `ts_max`. Rather I believe that `max` means elementwise maximum of two arrays (i.e., like `pmax(...)` in R and `np.maximum(...)` in Numpy) and have implemented it as such; same for `min(a, b)`. 
* `ts_argmax(x, days)`: on which day `ts_max(x, days)` occurred (also `ts_argmin(...)`) scaled to the interval [1/days,1]. For example, if window (*days*) is 10 days, and the max is in the most recent day, it will return 1.0; if the max is in the earliest day it will return 0.10.
* `ts_rank(x, days)`: the time series rank per asset on *x* over the the trailing *days*. Currently this is in the range [0,1], but should be [1/days,1].
* `sum(x, days)`: the sum per asset on *x* over the trailing *days*.
* `product(x, days)`: the product per asset on *x* over the trailing *days*.
* `stddev(x, days)`: the standard deviation per asset on *x* over the trailing *days*.
* `adv{days}`: the average daily **dollar** volume per asset over the trailing *days* (e.g., `adv20` gives the 20-day trailing average daily dollar volume).

The expression alpha parser produces `zipline` compatible `Pipeline` factor code. This implementation makes use of the `bottleneck` package which provides many `numpy`-style rolling aggregations, implemented in highly optimized compiled C code. The `bottleneck` package is distributed in binary form in the Anaconda Python distribution (see Installation below).

For example, the expression alpha "#9" from the paper

```
((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))
```

is compiled into a usable `Pipeline` factor, `e`, as

```python
e = (
	ExpressionAlpha('((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : (-1 * delta(close, 1))))).
	make_pipeline_factor().
	pipeline_factor(mask=universe)
)
```


The abstract snytax tree ("AST") can be visualized with `from lark.tree import pydot__tree_to_png; pydot__tree_to_png(e.tree, "alpha9.png")`:

<img src="https://user-images.githubusercontent.com/16124573/45169838-6e7e0f00-b1cc-11e8-9967-0c9d8bf70172.png" width="750">

This is quite helpful, in my opinion, to understand a third-party alpha like this. So what's happening? Looking top to bottom at each level, left to right: if zero is less than the minimum of the daily price change over the trailing five days (i.e., if the stock has gone **up** *every day* for the last five days), then the factor value is simpy the price change over the *most recent* day, which is a positive number by definition, and thus bets that positive momentum will continue. That branch should be pretty rare (meaning it would be rare for a stock to go up every day for five days in a row). Otherwise, we check if the max price change in the last 5 days is less than zero (i.e., the stock has gone **down** *every day* for the last 5 days), then the factor value again is just the price change over the *most recent day*, which is a negative number by definition. Thus if the stock has gone straight down for 5 days, the factor bets that it will continue. This should also be rare. Lastly, if neither of these two states exist, the factor value is just -1 times the last day's price change; i.e., a bet on mean reversion. Hence, by inspecting the parse tree like this, we can understand that this alpha is a momentum/mean-reversion switching factor; it assumes momentum will persist if the prior five days have moved in the same direction, otherwise it assumes mean-reversion will occur.

You can see the resuling `Pipeline` code (though this is not necessary to use the alpha in `run_pipeline`) with `print(e.pipeline_code)`:

```python
class ExprAlpha_1(CustomFactor):
    inputs = [USEP.close]
    window_length = 17

    def compute(self, today, assets, out, close):
        v0 = close - np.roll(close, 1, axis=0)
        v1 = bn.move_min(v0, window=5, min_count=1,  axis=0)
        v2 = np.less(0, v1)
        v3 = close - np.roll(close, 1, axis=0)
        v4 = close - np.roll(close, 1, axis=0)
        v5 = bn.move_max(v4, window=5, min_count=1,  axis=0)
        v6 = np.less(v5, 0)
        v7 = close - np.roll(close, 1, axis=0)
        v8 = close - np.roll(close, 1, axis=0)
        v9 = 1*v8
        v10 = -v9
        v11 = np.where(v6, v7, v10)
        v12 = np.where(v2, v3, v11)
        out[:] = v12[-1]
```

There is no compile-time optimization of the AST at all! What is happening is that the compiler walks down the AST and converts each node into a Python equivalent (`numpy`, `bottleneck`, and/or `pandas`) expression, keeping track of the call stack so that future references to prior calculations are correct. The resulting Python code is in the style of "three-address code". There is of course plenty of optimization which can be done.

Note that there is no reference implementation of the expression-style alpha syntax to test against and that there are many specific details lacking the paper. As such, this implementation makes some assumptions where necessary (as a simple example, the paper does not specify if `rank` is ascending or descending, however, it obviously should be ascending as a larger raw value should produce a larger numberical rank to keep the alpha vector *directly* proportional). This is experimental and I have created only a handful of tests.

### Using Your Own Data in Expression Alphas

It is also possible to use the "bring your own data" functionality provided by the `Factory` object in an expression alpha. This is done with one or more `factory` expressions. The syntax is

* `factory("<dataset>")`: where `"<dataset>"` is the name you would pass into the `Factory` object (for now assuming the data is in a column called "value"). Concretely, if you have a dataset, "sample", defined in the `data_sources.json` file, you can access it in an expression as:

```
(returns > 0) ? factory("sample") : -sum(returns, 5)
```

This compiles to the `Pipeline` factor as:

```python
class ExprAlpha_1(CustomFactor):
    inputs = [Returns(window_length=2), Factory["sample"].value]
    window_length = 7

    def compute(self, today, assets, out, returns, factory0):
        v0 = np.greater(returns, 0)
        v1 = pd.DataFrame(data=returns).rolling(
            window=5, center=False, min_periods=1).sum().values
        v2 = -v1
        v3 = np.where(v0, factory0, v2)
        out[:] = v3[-1]
```


## Installation

These install steps worked for me on Max OS X. Minimally you need a proper install of `zipline`. Zipline is built against certain version of `numpy` and `pandas` which can make it tricky. For example, if you want to use `scikit-learn` you have to compile it versus that `numpy` version specifically (needing `gcc` via Apple dev tools or via `brew`). Currently this package has been developed for Python 2.7. The install process that worked for me is as follows.

### Create Zipline Environment

```
conda create -n py27 python=2.7 anaconda
source activate py27
conda install -c Quantopian zipline=1.1.1
conda install pandas-datareader==0.2.1
conda install networkx==1.9.1
pip install scikit-learn --no-binary :all:
zipline ingest
```

### Install `alphatools`

```
pip install alphatools
```

This package is under very active development. For the time being, better is likely 

```
git clone https://github.com/marketneutral/alphatools
pip install -e alphatools
```

Note that when you run `zipline ingest` the security master is built from scratch and each `sid` is assigned at that time. You must map the `Sector`, `Industry` classifiers in this package **and all your own data** after every `zipline ingest`. You can map the `Sector` and `Industry` classifiers with

```
alphatools ingest
```

`zipline` requires a version of `blaze` which is not on PyPI. As such, you can get the compatible version with the following command. Note that this runs a `pip install` so make sure you have activated the environment.

```
alphatools get_blaze
```

You'll want to make the `py27` env available to Jupyter. To do this run

```
python -m ipykernel install --user --name py27 --display-name "Python 2.7 (py27)"
```


## A Word on Sector and Industry Classfiers Included

Sector and Industry data were scraped from Yahoo Finance on September 18, 2017 for the full Quandl WIKI universe at that time. The SIC and CIK codes were scraped from [Rank and Filed](http://rankandfiled.com/) on September 15, 2017. The classifiers built from this data assume that the codes have never and do never change (i.e., there is no concept of an asset being reclassified over time). **Be aware that there is lookahead bias in this** (e.g., a good example of why there is lookahead bias is with Corning, Inc. which is classified as a Technology/Electronic Components company in this dataset, but from 1851 to the 2000s(?) was actually classified as a boring Industrial glass company; the economic make up the company changed sometime in the early 1990s when optic fiber production became an important revenue driver and later with iPhone glass. At some point, the ICS providers changed the classification from "boring" to "high tech", but this was surely lagging the actual transformation of the company; hence...lookahead bias).

## A Word on Fundamental Data

Altough there is a `Fundamentals` factor included, there is no Fundamental data included in the package. This factor was built on top of the `DataFrameLoader` to get a `pandas.DataFrame` into a factor. I think I will deprecate this in favor of using the `Factory` object as described above. In the meantime, the `Fundamentals` pipeline factors can be built from `make_fundamentals.py` with your own data. Note that these factors use the `DataFrameLoader` which means the data must fit in memory. 

## Disclaimer

Though this is in the `LICENSE` file, it bears noting that this software is provided on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, including, without limitation, any warranties or conditions of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE

Additionally, nothing in this package constitutes investment advice. This package is a personal project and nothing in its functionality or examples is reflective of any past or current employer.

Lastly, there are no automated tests (or any significnat tests for that matter), no automated nightly build, no docstrings, or any other features associated with what you might consider a well supported open source package. 

## Contributing

I hope you enjoy this package. Please leave feedback, or better, contribute. If you are planning to make a PR, please get in touch with me before you do any work as I have a project plan. I am figuring this out as I go and could use help, especially with (in order)

- Incorporating `six` so that the package works with Python 3.x and Python 2.7
- Creating tests and using Travis CI on this repo
- Python packaging
- Dockerizing this thing so we can avoid the painful install process
