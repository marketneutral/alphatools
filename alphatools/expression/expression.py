from __future__ import print_function
import autopep8
import itertools
from lark import Lark, Transformer
from os import path
from scipy.stats import rankdata


class MyTransformer(Transformer):
    
    def __init__(self):
        self.cmdlist = []
        self.window = 2
        self.vcounter = itertools.count()
        self.stack = []
        
        self.imports = set()

        self.factory_counter = itertools.count()
        self.factories = dict()

        self.inputs = dict()

        
    def factory(self, items):
        self.imports.add('from alphatools.data import Factory')
        this_factory = self.factory_counter.next()
        self.stack.append('factory' + str(this_factory))
        self.factories[this_factory] = items[0]
        self.inputs['factory'+str(this_factory)] = 'Factory['+items[0]+'].value'
        
    def neg(self, items):
        term1 = self.stack.pop()
        thisv = self.vcounter.next()
        self.stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = -' + term1
        )

    def rank(self, items):
        self.imports.add("from scipy.stats import rankdata")
        term1 = self.stack.pop()
        v1 = self.vcounter.next()
        self.cmdlist.append(
            'v' + str(v1) + ' = np.apply_along_axis(rankdata, 1, ' + term1 +', method="ordinal")'
        )
        v2 = self.vcounter.next()
        self.stack.append('v' + str(v2))
        self.cmdlist.append(
            'v' + str(v2) + ' = np.divide(v'+str(v1)+'.astype(float), np.sum(~np.isnan(v'+str(v1)+'), axis=1).reshape(v'+str(v1)+'.shape[0], 1))'
        )
        
    
#    def close(self, items):
#        thisv = self.vcounter.next()
#        self.stack.append('v' + str(thisv))
#        self.cmdlist.append(
#            'v' + str(thisv) + ' = close'
#        )

    def cap(self, items):
        thisv = self.vcounter.next()
        self.stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = 1.0'
        )

    def number(self, items):
        #import pdb; pdb.set_trace()
        self.stack.append(str(items[0].value))
        pass

    def close(self, items):
        self.imports.add('from zipline.pipeline.data import USEquityPricing as USEP')
        self.inputs['close'] = 'USEP.close'
        self.stack.append('close')

    def high(self, items):
        self.imports.add('from zipline.pipeline.data import USEquityPricing as USEP')
        self.inputs['high'] = 'USEP.high'
        self.stack.append('high')

    def low(self, items):
        self.imports.add('from zipline.pipeline.data import USEquityPricing as USEP')
        self.inputs['low'] = 'USEP.low'
        self.stack.append('low')
        
    def volume(self, items):
        self.imports.add('from zipline.pipeline.data import USEquityPricing as USEP')
        self.inputs['volume'] = 'USEP.volume'
        self.stack.append('volume')

    def vwap(self, items):
        self.imports.add('from zipline.pipeline.data import USEquityPricing as USEP')
        self.inputs['close'] = 'USEP.close'
        self.inputs['opens'] = 'USEP.open'
        self.inputs['high'] = 'USEP.high'
        self.inputs['low'] = 'USEP.low'
        
        thisv = self.vcounter.next()
        self.stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = (close + (opens + high + low)/3)/2'
        )

    def adv(self, items):
        self.imports.add('from zipline.pipeline.data import USEquityPricing as USEP')
        self.inputs['close'] = 'USEP.close'
        self.inputs['volume'] = 'USEP.volume'
        thisv = self.vcounter.next()
        self.stack.append('v' + str(thisv))
        self.window = max([self.window, int(items[0])+2])
        self.cmdlist.append(
            'v' + str(thisv) + ' = bn.move_mean(np.multiply(close, volume), window=' + items[0] + ', min_count=1, axis=0)'
        )
#    def opens(self, items):
#        thisv = self.vcounter.next()
#        self.stack.append('v' + str(thisv))
#        self.cmdlist.append(
#            'v' + str(thisv) + ' = opens'
#        )

    def opens(self, items):
        self.imports.add('from zipline.pipeline.data import USEquityPricing as USEP')
        self.inputs['opens'] = 'USEP.open'
        self.stack.append('opens')
                
    def div(self, items):
        term2 = self.stack.pop()
        term1 = self.stack.pop()
        thisv = self.vcounter.next()
        self.stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = ' + term1 + ' / ' + term2
        )

    def min(self, items):
        # TODO: check that this is parallel min 
        term2 = self.stack.pop()
        term1 = self.stack.pop()
        thisv = self.vcounter.next()
        self.stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = np.minimum('+term1 + ', ' + term2+')'
        )
        
    def max(self, items):
        # TODO: check that this is parallel max
        # paper says this is == ts_min, but that doesn't parse for alpha 71
        term2 = self.stack.pop()
        term1 = self.stack.pop()
        thisv = self.vcounter.next()
        self.stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = np.maximum('+term1 + ', ' + term2+')'
        )
        
    def powerof(self, items):
        """ Element-wise power """

        term2 = self.stack.pop()
        term1 = self.stack.pop()
        thisv = self.vcounter.next()
        self.stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = np.power(' + term1 + ', ' + term2 + ')'
        )

    def signedpower(self, items):
        """ np.sign(term1)*np.power(np.abs(term1), term2)  """

        term2 = self.stack.pop()
        term1 = self.stack.pop()
        thisv = self.vcounter.next()
        self.stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = np.sign('+term1+')*np.power(np.abs(' + term1 + '), ' + term2 + ')'
        )
        

    def minus(self, items):
        term2 = self.stack.pop()
        term1 = self.stack.pop()
        thisv = self.vcounter.next()
        self.stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = ' + term1 + ' - ' + term2
        )

    def plus(self, items):
        term2 = self.stack.pop()
        term1 = self.stack.pop()
        thisv = self.vcounter.next()
        self.stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = ' + term1 + ' + ' + term2
        )

    def mult(self, items):
        term2 = self.stack.pop()
        term1 = self.stack.pop()
        thisv = self.vcounter.next()
        self.stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = ' + term1 + '*' + term2
        )

    def log(self, items):
        term1 = self.stack.pop()
        thisv = self.vcounter.next()
        self.stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = np.log(' + term1 + ')'
        )

    def abs(self, items):
        term1 = self.stack.pop()
        thisv = self.vcounter.next()
        self.stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = np.abs(' + term1 + ')'
        )

    def sign(self, items):
        term1 = self.stack.pop()
        thisv = self.vcounter.next()
        self.stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = np.sign(' + term1 + ')'
        )
        
    def scale(self, items):
        # TODO: 101 paper says scaled sum(abs)==a; silent on mean
        term1 = self.stack.pop()
        thisv = self.vcounter.next()
        self.stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = np.apply_along_axis(lambda x: (x - np.nanmean(x))/np.nansum(np.abs(x - np.nanmean(x))), 1, ' + term1 +')'
        )
        
    def mult(self, items):
        term2 = self.stack.pop()
        term1 = self.stack.pop()
        thisv = self.vcounter.next()
        self.stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = ' + term1 + '*' + term2
        )
        
    def greaterthan(self, items):
        term2 = self.stack.pop()
        term1 = self.stack.pop()
        thisv = self.vcounter.next()
        self.stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = np.greater(' + term1 + ', ' + term2 + ')'
        )

    def lessthan(self, items):
        term2 = self.stack.pop()
        term1 = self.stack.pop()
        thisv = self.vcounter.next()
        self.stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = np.less(' + term1 + ', ' + term2 + ')'
        )

    def equals(self, items):
        # TODO: do we want np.isclose or np.allcose?
        term2 = self.stack.pop()
        term1 = self.stack.pop()
        thisv = self.vcounter.next()
        self.stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = np.isclose(' + term1 + ', ' + term2 + ')'
        )

    def logicalor(self, items):
        term2 = self.stack.pop()
        term1 = self.stack.pop()
        thisv = self.vcounter.next()
        self.stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = np.logical_or(' + term1 + ', ' + term2 + ')'
        )

    def ternary(self, items):
        term3 = self.stack.pop()
        term2 = self.stack.pop()
        term1 = self.stack.pop()
        thisv = self.vcounter.next()
        self.stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = np.where(' + term1 + ', ' + term2 + ', ' + term3 + ')'
        )
        
    def returns(self, items):
        self.imports.add("from zipline.pipeline.factors import Returns")
        self.inputs['returns'] = 'Returns(window_length=2)'
        self.stack.append('returns')
        #thisv = self.vcounter.next()
        #self.window = self.window+1
        #self.stack.append('v' + str(thisv))
        #self.cmdlist.append(
        #    'v' + str(thisv) + ' = np.log(close/np.roll(close, 1, axis=0))'
        #)

        
    def delta(self, items):
        term1 = self.stack.pop()
        thisv = self.vcounter.next()
        self.window = self.window+int(items[1])
        self.stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = '+term1+' - np.roll(' + term1 + ', ' + items[1] + ', axis=0)'
        )

    def delay(self, items):
        term1 = self.stack.pop()
        thisv = self.vcounter.next()
        self.window = self.window+int(items[1])
        self.stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = np.roll(' + term1 + ', ' + items[1] + ', axis=0)'
        )
        
    def ts_max(self, items):
        v1 = self.stack.pop()
        thisv = self.vcounter.next()
        self.window = self.window + int(items[1])
        self.stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = bn.move_max(' + v1 + ', window=' + items[1] + ', min_count=1,  axis=0)'
        )

    def ts_min(self, items):
        v1 = self.stack.pop()
        thisv = self.vcounter.next()
        self.window = self.window + int(items[1])
        self.stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = bn.move_min(' + v1 + ', window=' + items[1] + ', min_count=1,  axis=0)'
        )

    def ts_argmax(self, items):
        """
        The behavior of `move_argmax` and associated functions in Numpy
        and Bottleneck is that they index based on the shape of the array.
        In this case the time increases along the 0 axis so, if window is
        10 days, and the max is in the most recent day, it will return 9;
        If the max is in the earliest day it will return zero. I add "1" to
        this imagining a mutiplier, and do not want zero to kill values.
        It is then rescaled to the interval (0,1] to match the `rank` style.
        """
        v1 = self.stack.pop()
        thisv = self.vcounter.next()
        self.window = self.window + int(items[1])
        self.stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = (1. + bn.move_argmax(' + v1 + ', window=' + items[1] + ', min_count=1,  axis=0))/' + items[1]
        )

    def ts_argmin(self, items):
        v1 = self.stack.pop()
        thisv = self.vcounter.next()
        self.window = self.window + int(items[1])
        self.stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = (1. + bn.move_argmin(' + v1 + ', window=' + items[1] + ', min_count=1,  axis=0))/' + items[1]
        )

    def ts_rank(self, items):
        # Returns ranks 1-N; largest value is rank N
        # `bn.move_rank` returns values in the range -1 to 1.0, so we add 1
        # to get 0-2 and then divide by 2.0 to get [0,1]
        # note that we want [1/N, 1]
        v1 = self.stack.pop()
        thisv = self.vcounter.next()
        self.window = self.window + int(items[1])
        self.stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = (1. + bn.move_rank(' + v1 + ', window=' + items[1] + ', min_count=1,  axis=0))/2.0'
        )
        
    def stddev(self, items):
        # check that the day is what we want
        v1 = self.stack.pop()
        thisv = self.vcounter.next()
        self.window = self.window + int(items[1])
        self.stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = bn.move_std(' + v1 + ', window=' + items[1] + ', min_count=1,  axis=0)'
        )

    def sum(self, items):
        v1 = self.stack.pop()
        thisv = self.vcounter.next()
        self.window = self.window + int(items[1])
        self.stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = pd.DataFrame(data='+v1+').rolling(window='+items[1]+', center=False, min_periods=1).sum().values'
        )

    def product(self, items):
        v1 = self.stack.pop()
        thisv = self.vcounter.next()
        self.window = self.window + int(items[1])
        self.stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = pd.DataFrame(data='+v1+').rolling(window='+items[1]+', center=False, min_periods=1).apply(lambda x: np.prod(x)).values'
        )

    def correlation(self, items):
        v2 = self.stack.pop()
        v1 = self.stack.pop()
        thisv = self.vcounter.next()
        self.window = self.window + int(items[2])
        self.stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = pd.DataFrame('+v1+').rolling(window='+items[2]+', min_periods='+items[2]+').corr(other=pd.DataFrame('+v2+')).values'
        )

    def covariance(self, items):
        v2 = self.stack.pop()
        v1 = self.stack.pop()
        thisv = self.vcounter.next()
        self.window = self.window + int(items[2])
        self.stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = pd.DataFrame('+v1+').rolling(window='+items[2]+', min_periods='+items[2]+').cov(other=pd.DataFrame('+v2+')).values'
        )
        
    def decay_linear(self, items):
        v1 = self.stack.pop()
        thisv = self.vcounter.next()
        days = int(items[1])
        self.window = self.window + days
        v2 = 'v'+str(thisv)
        self.cmdlist.append(
            v2 + ' = (np.arange(' + items[1] + ')+1.)/np.sum(np.arange(' + items[1]+ ')+1.)'
        )
        thisv = self.vcounter.next()
        self.stack.append('v' + str(thisv))

        self.cmdlist.append(
            'v' + str(thisv) + ' = pd.DataFrame(data='+v1+').rolling(window='+items[1]+', center=False, min_periods='+items[1]+').apply(lambda x: (x*'+v2+').sum()).values'
        )

    def indneutralize(self, items):
        """
        De-means a data matrix, data, DxN, D days in rows x N stocks in
        columns by group means.

        The group means come from Pipeline Classifiers: Sector() and 
        SubIndustry(). These are integer values per stock; -1 for missing.

        The Classifier produces a matrix window_lengthxN. We need the last
        slice of this, assuming that the data is constant per day.

        We set up a factor indicator matrix, OHE, like a one-hot-encoded
        matrix.

        # set up OHE matrix; add 1 so that missing now == 0
        OHE = np.zeros(N, classifier.max()+2)
        OHE[np.arange(N), classifier[-1] + 1) = 1

        # The per day (rows) by per industry (columns) mean is
        per_day_per_ind_mean = data.dot(OHE)/OHE.sum(axis=0)
     
        # The per day (rows) per *asset* (column) mean then is
        per_day_per_asset_ind_mean = per_day_per_ind_mean.dot(OHE.T)

        Finally, the de-meaned data matrix is simply calculated as

        data = data - per_day_per_asset_ind_mean
        """
        self.imports.add("from alphatools.ics import Sector, SubIndustry")
        self.inputs['sector'] = 'Sector()'
        self.inputs['subindustry'] = 'SubIndustry()'

        groupmap = {
            'IndClass.subindustry': 'subindustry',
            'IndClass.sector': 'sector',
            'IndClass.industry': 'subindustry',
        }
        
        v1 = self.stack.pop()
        if len(items)<2:
            groupby = 'IndClass.subindustry'
        else:
            groupby = str(items[1])

        group_label = groupmap[groupby]
        
        # set up ICS matrix (like one-hot-encoded matrix); we add 1 to the
        # ics scheme bc -1 is a missing, so increment all by 1
        ohe = 'v' + str(self.vcounter.next())
        self.cmdlist.append(
            ohe + ' = np.zeros(('+group_label+'.shape[1], '+group_label+'.max()+2))'
        )
        self.cmdlist.append(
            ohe + '[np.arange('+group_label+'.shape[1]), '+group_label+'[-1] + 1] = 1'
        )

        # get industry mean, per industry on columns, per day on rows
        # and the dot(ohe.T) gives per stock industry mean
        ind_mean = 'v' + str(self.vcounter.next())
        self.cmdlist.append(
            ind_mean + ' = (np.nan_to_num('+v1+'.dot('+ohe+')/'+ohe+'.sum(axis=0))).dot('+ohe+'.T)'
        )
        
        thisv = self.vcounter.next()
        self.stack.append('v' + str(thisv))
        # subtract the per stock industry mean
        self.cmdlist.append(
            'v' + str(thisv) + ' = '+v1+' - '+ind_mean
        )
        
        
    def transform(self, tree):
        self._transform_tree(tree)
        v1 = self.stack.pop()
        self.cmdlist.append(
            'out[:] = ' + v1 + '[-1]'
        )
        return self
        #return ["window_length = "+str(self.window)] + self.cmdlist


class ExpressionAlpha():

    def __init__(self, expr_string):
        self.expr_string = expr_string
        self.code = ""
        fname = path.join(path.dirname(__file__), 'expression.lark')
        with open(fname, 'r') as grammar_file:
            self.grammar = grammar_file.read()

    def make_pipeline_factor(self):
        self.parse()
        self.transform()
        self.generate_pipeline_code()
        exec(self.imports, globals(), globals())
        exec(self.pipeline_code, globals(), globals())
        self.pipeline_factor = ExprAlpha_1
        return self
    
    def parse(self):
        my_parser = Lark(self.grammar, start='value')
        self.tree = my_parser.parse(self.expr_string)
        return self

    def transform(self):
        self.transformed = MyTransformer().transform(self.tree)
        return self

    def generate_pipeline_code(self):
        raw_np_list = \
            ["window_length = "+str(self.transformed.window)] + \
            self.transformed.cmdlist
        raw_imports = \
            self.transformed.imports

        (data_names, factor_names) = zip(*self.transformed.inputs.iteritems())
        
        self.imports = ['{0}\n'.format(imp) for imp in raw_imports]
        self.imports.append("from zipline.pipeline.factors import CustomFactor\n")
        self.imports.append("import numpy as np\n")
        self.imports.append("import bottleneck as bn\n")
        self.imports.append("import pandas as pd\n")
        self.imports = ["from __future__ import division\n"] + \
            self.imports
        
        self.code = ["class ExprAlpha_1(CustomFactor):"]

        self.code.append("    inputs = [" + ', '.join(factor_names) + "]")
        self.code.append('    {0}'.format(raw_np_list[0]))
        self.code.append("    def compute(self, today, assets, out, " + ', '.join(data_names) + "):")
        lst = ['        {0}'.format(elem) for elem in raw_np_list]

        self.code = self.code + lst[1:]

        self.imports = ''.join(self.imports)
        
        self.code_string = '\n'.join(self.code)
        self.pipeline_code = autopep8.fix_code(self.code_string)
        return self

if __name__ == '__main__':
    e = ExpressionAlpha('close/delay(opens,1)')
    e.to_pipeline()
    print(e.pipeline_code)
