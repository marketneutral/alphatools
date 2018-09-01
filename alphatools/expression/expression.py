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
    
    def close(self, items):
        thisv = self.vcounter.next()
        self.stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = close'
        )
        
    def opens(self, items):
        thisv = self.vcounter.next()
        self.stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = opens'
        )

    def div(self, items):
        term2 = self.stack.pop()
        term1 = self.stack.pop()
        thisv = self.vcounter.next()
        self.stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = ' + term1 + '/' + term2
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
            'v' + str(thisv) + ' = np.nansum(' + v1 + '[-' + items[1] +':, :], axis=0)'
        )
        
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

    def to_pipeline(self):
        raw_np_list = self._parse()
        
        self.code = ['class ExprAlpha_1(CustomFactor):\n']
        self.code.append("    inputs = [USEP.open, USEP.high, USEP.low, USEP.close]\n")
        self.code.append('    {0}'.format(raw_np_list[0]))
        self.code.append("    def compute(self, today, assets, out, opens, high, low, close):\n")
        lst = ['        {0}'.format(elem) for elem in raw_np_list]

        self.code = self.code + lst[1:]
        self.code_string = '\n'.join(self.code)
        self.final = autopep8.fix_code(self.code_string)

    def _parse(self):
        my_parser = Lark(self.grammar, start='value')
        self.tree = my_parser.parse(self.expr_string)
        return(MyTransformer().transform(self.tree))

    def _make_code(self):
        pass

if __name__ == '__main__':
    e = ExpressionAlpha('close/delay(opens,1)')
    e.to_pipeline()
    print(e.final)
