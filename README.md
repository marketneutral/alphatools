# alphatools

This allows you to run a Jupyter notebook research sandbox for factor research.

For example, with a Jupyter notebook, you can

```python

from alphatools.alphatools import run_pipeline
from alphatools.ics.ics_scheme import Sector
from alphatools.fundamentals.fundamentals import Fundamentals
from zipline.pipeline.data import USEquityPricing as USEP
from zipline.pipeline.factors import Returns, AverageDollarVolume
from zipline.pipeline import Pipeline

universe = AverageDollarVolume(window_length=120).top(500)
my_factor = (
    -Returns(mask=universe, window_length=5).
    demean(groupby=Sector()).
    rank()
)

p = Pipeline(screen=universe)
p.add(my_factor, '5d_MR_Sector_Neutral_Rank')
p.add(Sector(), 'Sector')
p.add(Fundamentals.MarketCap.latest.zscore(), 'MCAP')
p.add(Fundamentals.PriceToBook.latest.zscore(), 'PB')

start_date = '2017-01-04'
end_date = '2017-12-28'

df = run_pipeline(p, start_date, end_date)
```

## Installation

These install steps worked for me on Max OS X. Minimally you need a proper install of `zipeline`. Zipline is built against certain version of `numpy` and `pandas` which can make it tricky. For example, if you want to use `scikit-learn` you have to compile it versus that `numpy` version specifically (needing `gcc` via Apple dev tools or via `brew`). Currently this package has been developed for Python 2.7. The install process that worked for me is

```
conda create -n py27 python=2.7 anaconda
source activate py27
conda install -c Quantopian zipline=1.1.1
conda install pandas-datareader==0.2.1
conda install networkx==1.9.1
pip install blaze
pip install scikit-learn --no-binary
pip install ipykernel
python -m ipykernel install --user --name py27 --display-name "Python 2.7 (py27)"
zipline ingest
pip install alphatools
pip install alphalens
python ics_scheme.py 
```

Note that when you run `zipline ingest` the security master is built from scratch and each `sid` is assigned at that time. You must map the `Sector`, `Industry`, etc. classifiers in this package after every `zipline ingest` with `python ics_scheme.py`.

## Data

Sector and Industry data were scraped from Yahoo Finance on September 18, 2017 for the full Quandl WIKI universe at that time. The SIC and CIK codes were scraped from [Rank and Filed](http://rankandfiled.com/). The classifiers built from this data assume that the codes have never and do never change (i.e., there is no concept of an asset being reclassified over time). **Be aware that there is lookahead bias in this** (e.g., a good example of why there is lookahead bias is with Corning, Inc. which is classified as a Technology/Electronic Components company in this dataset, but from 1851 to the 2000s(?) was actually classified as a boring Industrial glass company; the economic make up the company changed sometime in the early 1990s when optic fiber production became an important revenue driver and later with iPhone glass. At some point, the ICS providers changed the classification from "boring" to "high tech", but this was surely lagging the actual transformation of the company; hence...lookahead bias). There is no Fundamental data included in the package; the `Fundamentals` pipeline factors can be built from `make_fundamentals.py` with your own data. Note that these factors use the `DataFrameLoader` which means the data must fit in memory. Alternatively you can see the example of using the `BlazeLoader` in the `notebooks` directory.

## Disclaimer

Though this is in the `LICENSE` file, it bears noting that this software is provided on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied, including, without limitation, any warranties or conditions of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A PARTICULAR PURPOSE

Additionally, nothing in this package constitutes investment advice. This package is a personal project and nothing in its functionality or examples is reflective of any past or current employer.

Lastly, there are no automated tests (or any tests for that matter), no docstrings, or any other features associated with what you might consider a well supported open source package. 

## Contributing

I hope you enjoy this package. Please leave feedback, or better, contribute. If you are planning to make a PR, please get in touch with me before you do any work. Things that would be awesome to work on (in order):

- incorporating `six` so that the package works with Python 3.x and Python 2.7
- Dockerizing this thing so we can avoid the painful install process
