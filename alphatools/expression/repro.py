from __future__ import print_function
import itertools
from lark import Lark, Transformer
from scipy.stats import rankdata

grammar = r"""

    ?value: "(" value ")"
          | log
          | ts_max
          | ts_min
          | mat1
          | mat2
          | delay
          | div
          | mult
          | neg
          | rank
          | sum
          | SIGNED_NUMBER
          | ESCAPED_STRING


    list : "(" [value ("," value)*] ")"
    delay: "delay" "(" value "," SIGNED_NUMBER ")"
    mat1: "mat1"
    mat2: "mat2"
    ts_max: "ts_max" "(" value "," SIGNED_NUMBER ")"
    ts_min: "ts_min" "(" value "," SIGNED_NUMBER ")"
    div: value "/" value
    mult: value "*" value
    log: "log" "(" value ")"
    neg: "-" value
    rank: "rank" "(" value ")"
    sum: "sum" "(" value "," SIGNED_NUMBER ")"

    %import common.SIGNED_NUMBER
    %import common.ESCAPED_STRING
    %import common.WS
    %ignore WS

    """


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
    
    def mat1(self, items):
        thisv = self.vcounter.next()
        self.stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = mat1'
        )
        
    def mat2(self, items):
        thisv = self.vcounter.next()
        self.stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = mat2'
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
        
        return ['window='+str(self.window)] + self.cmdlist


my_parser = Lark(grammar, start='value')

text = "ts_max(mat1/mat2, 20) / delay(mat1, 5)"
text = "-log((mat1*mat1)/(ts_max(mat1,20)*mat1))"
text = "rank(log(mat1/delay(mat1,1)))"
text = "-sum(log(mat1/delay(mat1,1)),20)"
tree = my_parser.parse(text)
npcmds = MyTransformer().transform(tree)

import numpy as np

mat1 = np.array(np.random.random(200)).reshape(20,10)
mat2 = np.array(np.random.random(200)).reshape(20,10)
out = np.zeros(shape=(1,10))

for cmd in npcmds:
    exec(cmd)
