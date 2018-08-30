import blaze as bz
import json
from datashape import dshape

from alphatools.research import loaders, blaze_loader
from zipline.data import bundles
from zipline.utils.calendars import get_calendar
from zipline.pipeline.loaders.blaze import from_blaze

from os import path

this_file = path.dirname(__file__)

with open(path.join(this_file, 'factory/data_sources.json')) as f:
    data_sources = json.load(f)

Factory = {}

for source in data_sources.keys():
    loc = data_sources[source]['url']
    shape = dshape(data_sources[source]['schema'])

    loc = path.expandvars(loc)

    expr = bz.data(
        loc,
        dshape=shape
    )
    
    # create the DataSet
    ds = from_blaze(
        expr,
        no_deltas_rule='ignore',
        no_checkpoints_rule='ignore',
        loader=blaze_loader
    )
    Factory = {source: ds}
