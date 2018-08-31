from __future__ import print_function
import itertools
from lark import Lark, Transformer
import numpy as np

# define the EBNF grammar
grammar = r"""

    ?value: list
          | delay
          | close_alone
          | close
          | div
          | SIGNED_NUMBER
          | ESCAPED_STRING

    list : "(" [value ("," value)*] ")"
    delay: "delay" "(" value "," SIGNED_NUMBER ")"
    close_alone: "close_alone"
    close: "close" "," [value ("," value)*]
    div: value "/" value

    %import common.SIGNED_NUMBER
    %import common.ESCAPED_STRING
    %import common.WS
    %ignore WS

    """

class MyTransformer(Transformer):
    vcounter = itertools.count()
    np.random.seed(100)
    prices = np.random.random(100)

    def __init__(self):
        self.nplist = []

    def list(self):
        pass

    def close(self, items):
        thisv = self.vcounter.next()
        self.nplist.append("v" + str(this) + " = prices") 
    
    def close_alone(self, items):
        thisv = self.vcounter.next()
        self.nplist.append("v" + str(thisv) + " = prices[-1]")

    def div(self, items):
        thisv = self.vcounter.next()
        self.nplist.append("v" + str(thisv) + " = v" + str(thisv - 2) + "/v" + str(thisv-1))

    def delay(self, items):
        print(items)
        thisv = self.vcounter.next()
        self.nplist.append("v" + str(thisv) + " = v" + str(thisv -1) + "[-" + items[1] + "]")

    def transform(self, tree):
        self._transform_tree(tree)
        return self.nplist


parser = Lark(grammar, start='value')

text = "close/delay(close, 5)"
tree = parser.parse(text)
print(tree.pretty())
print(MyTransformer().transform(tree))
