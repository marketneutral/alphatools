from __future__ import print_function
import autopep8
import itertools
from lark import Lark, Transformer
from os import path
from scipy.stats import rankdata


class MyTransformer(Transformer):
    vcounter = itertools.count()
    stack = []
    window = 2
    
    def __init__(self):
        self.cmdlist = []

    def neg(self, items):
        term1 = self.stack.pop()
        thisv = self.vcounter.next()
        self.stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = -' + term1
        )

    def rank(self, items):
        term1 = self.stack.pop()
        thisv = self.vcounter.next()
        self.stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = np.apply_along_axis(rankdata, 1, ' + term1 +')'
        )
    
#    def close(self, items):
#        thisv = self.vcounter.next()
#        self.stack.append('v' + str(thisv))
#        self.cmdlist.append(
#            'v' + str(thisv) + ' = close'
#        )
    def number(self, items):
        import pdb; pdb.set_trace()
        self.stack.append(str(items[0].value))
        pass

    def close(self, items):
        #import pdb; pdb.set_trace()
        self.stack.append('close')

    def high(self, items):
        self.stack.append('high')

    def low(self, items):
        self.stack.append('low')
        
    def volume(self, items):
        self.stack.append('volume')
        
#    def opens(self, items):
#        thisv = self.vcounter.next()
#        self.stack.append('v' + str(thisv))
#        self.cmdlist.append(
#            'v' + str(thisv) + ' = opens'
#        )

    def opens(self, items):
        self.stack.append('opens')
                
    def div(self, items):
        term2 = self.stack.pop()
        term1 = self.stack.pop()
        thisv = self.vcounter.next()
        self.stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = ' + term1 + ' / ' + term2
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
        
    def delay(self, items):
        term1 = self.stack.pop()
        thisv = self.vcounter.next()
        self.window = self.window+int(items[1])
        self.stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = np.roll(' + term1 + ', ' + items[1] + ')'
        )

    def returns(self, items):
        self.stack.append('returns')
        #thisv = self.vcounter.next()
        #self.window = self.window+1
        #self.stack.append('v' + str(thisv))
        #self.cmdlist.append(
        #    'v' + str(thisv) + ' = np.log(close/np.roll(close, 1))'
        #)

        
    def delta(self, items):
        term1 = self.stack.pop()
        thisv = self.vcounter.next()
        self.window = self.window+int(items[1])
        self.stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = '+term1+' - np.roll(' + term1 + ', ' + items[1] + ')'
        )

        
    def ts_max(self, items):
        v1 = self.stack.pop()
        thisv = self.vcounter.next()
        self.window = self.window + int(items[1])
        self.stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = np.nanmax(' + v1 + '[-' + items[1] +':, :], axis=0)'
        )

    def ts_min(self, items):
        v1 = self.stack.pop()
        thisv = self.vcounter.next()
        self.window = self.window + int(items[1])
        self.stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = np.nanmin(' + v1 + '[-' + items[1] +':, :], axis=0)'
        )

    def sum(self, items):
        v1 = self.stack.pop()
        thisv = self.vcounter.next()
        self.window = self.window + int(items[1])
        self.stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = pd.DataFrame(data='+v1+').rolling(window='+items[1]+', center=False, min_periods=1).sum().values'
        )
        
        
        #self.cmdlist.append(
        #    'v' + str(thisv) + ' = np.nansum(' + v1 + '[-' + items[1] +':, :], axis=0)'
        #)
        
    def transform(self, tree):
        self._transform_tree(tree)
        v1 = self.stack.pop()
        self.cmdlist.append(
            'out[:] = ' + v1 + '[-1]'
        )
        
        return ["window_length = "+str(self.window)] + self.cmdlist


class ExpressionAlpha():

    def __init__(self, expr_string):
        self.expr_string = expr_string
        self.code = ""
        fname = path.join(path.dirname(__file__), 'expression.lark')
        with open(fname, 'r') as grammar_file:
            self.grammar = grammar_file.read()

    def pipeline_factor(self):
        self.parse()
        self.transform()
        self.generate_pipeline_code()
        self.make_pipeline_factor()
        return self.pipeline_factor
    
    def parse(self):
        my_parser = Lark(self.grammar, start='value')
        self.tree = my_parser.parse(self.expr_string)

    def transform(self):
        self.raw_np_list = MyTransformer().transform(self.tree)

    def make_pipeline_factor(self):
        exec(self.imports, globals(), globals())
        exec(self.pipeline_code, globals(), globals())
        self.pipeline_factor = ExprAlpha_1
        
    def generate_pipeline_code(self):
        raw_np_list = self.raw_np_list

        self.imports = ["from zipline.pipeline.data import USEquityPricing as USEP\n"]
        self.imports.append("from zipline.pipeline.factors import CustomFactor, Returns\n")
        self.imports.append("import numpy as np\n")
        self.imports.append("import pandas as pd\n")
        self.imports.append("from scipy.stats import rankdata\n\n")
        self.code = ["class ExprAlpha_1(CustomFactor):"]
        self.code.append("    inputs = [Returns(window_length=2), USEP.open, USEP.high, USEP.low, USEP.close, USEP.volume]")
        self.code.append('    {0}'.format(raw_np_list[0]))
        self.code.append("    def compute(self, today, assets, out, returns, opens, high, low, close, volume):")
        lst = ['        {0}'.format(elem) for elem in raw_np_list]

        self.code = self.code + lst[1:]

        self.imports = ''.join(self.imports)
        
        self.code_string = '\n'.join(self.code)
        self.pipeline_code = autopep8.fix_code(self.code_string)


if __name__ == '__main__':
    e = ExpressionAlpha('close/delay(opens,1)')
    e.to_pipeline()
    print(e.pipeline_code)
