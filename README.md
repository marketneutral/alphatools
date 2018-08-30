![](https://user-images.githubusercontent.com/16124573/44810782-d0bd8b00-aba0-11e8-81f3-e4fe042c481d.png)


This package provides convenience functions to help make the alpha factor research process more accessible. The convenience functions sit on top of [zipline]() and, specifically, the `Pipeline` cross-sectional classes and functions in that package. `alphatools` allows you to `run_pipeline` in a Jupyter notebook local to you and supports the easy creation of `Pipeline` factors **at runtime** on arbitrary data sources. In other words, just expose the endpoint for data sitting somewhere, specify the schema, and...it's available for use in `Pipeline`!

For example, with `alphatools`, in a Jupyter notebook, you can

```python
from alphatools.research import run_pipeline
from alphatools.ics import Sector
from alphatools.data import Factory
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
p.add(Factory['my_special_data'].value.latest.zscore(), 'PB')

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
		"url": "postgresql://user:pass@hostname::my-table-name",
		"schema": "var * {asof_date: datetime, sid: int64, price_to_book: float64}"
}
```

The `schema` is specified in the `dshape` DSL from the package `datashape` with docs [here](). The magic happens via the `blaze/datashape/odo` stack. You can specify the `url` to a huge variety of source formats including `json`, `csv`, PostgreSQL tables, MongoDB collections, `bcolz`, Microsoft Excel(!?), `.gz` compressed files, collections of files (e.g., `myfiles_*.csv`), and remote locations like Amazon S3 and a Hadoop Distributed File System. To me, the `odo` [docs](http://odo.pydata.org/en/latest/uri.html) are the clearest on this.

Note that this data must be mapped to the `sid` as mapped by `zipline ingest`. Also, the data date my be in a column titled `asof_date`. (TODO: add `alphatools map <data_source> <output>`). You can then access this data like

```python
from alphatools.data import Factory
	:
	:
	:
	
my_factor = Factory['my_database_data'].price_to_book.latest.rank()
p.add(my_factor)
```

This functionality should allow you to use new data in research very quickly with the absolute minimal amount of data engineering. For example, commercial risk model providers often provide a single file per day for factor loadings (e.g., `data_yyyymmdd_fac.csv`). After `sid` mapping and converting the date column name to `asof_date`,  this data can be immediately available in `Pipeline` by putting a `url` in `data_sources.json` like `"url": "/path/to/dir/data_*_fac.csv"`, and `schema` like `"var * {asof_date: datetime, sid: int64, VALUE: float64, MOMENTUM: float64, ST_REVERSAL: float64 ..."`.

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
pip intstall -e alphatools
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

Lastly, there are no automated tests (or any tests for that matter), no docstrings, or any other features associated with what you might consider a well supported open source package. 

## Contributing

I hope you enjoy this package. Please leave feedback, or better, contribute. If you are planning to make a PR, please get in touch with me before you do any work as I have a project plan. I am figuring this out as I go and could use help, especially with (in order)

- Incorporating `six` so that the package works with Python 3.x and Python 2.7
- Creating tests and using Travis CI on this repo
- Python packaging
- Dockerizing this thing so we can avoid the painful install process
