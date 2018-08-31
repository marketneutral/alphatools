from __future__ import print_function
import itertools
from lark import Lark, Transformer

grammar = r"""

    ?value: list
          | max
          | mat1
          | mat2
          | lag
          | div
          | max
          | SIGNED_NUMBER
          | ESCAPED_STRING

    list : "(" [value ("," value)*] ")"
    lag: "lag" "(" value "," SIGNED_NUMBER ")"
    mat1: "mat1"
    mat2: "mat2"
    max: "max" "(" value ")"
    div: value "/" value

    %import common.SIGNED_NUMBER
    %import common.ESCAPED_STRING
    %import common.WS
    %ignore WS

    """

class MyTransformer(Transformer):
    vcounter = itertools.count()

    def __init__(self):
        self.nplist = []

    def list(self):
        pass

    def mat1(self, items):
        thisv = self.vcounter.next()
        self.nplist.append(
            "v" + str(thisv) + " = mat1[-1]"
        )
        
    def mat2(self, items):
        thisv = self.vcounter.next()
        self.nplist.append(
            "v" + str(thisv) + " = mat2[-1]"
        )

    def div(self, items):
        thisv = self.vcounter.next()
        self.nplist.append(
            "v" + str(thisv) + " = v" + str(thisv - 2) + "/v" + str(thisv-1)
        )

    def lag(self, items):
        thisv = self.vcounter.next()
        self.nplist.append(
            "v" + str(thisv) + " = v" + str(thisv -1) + "[-" + items[1] + "]"
        )

    def max(self, items):
        thisv = self.vcounter.next()
        self.nplist.append(
            "v" + str(thisv) + " = np.max(v" + str(thisv-1) + ", axis=0)"
        )

    def transform(self, tree):
        self._transform_tree(tree)
        return self.nplist

my_parser = Lark(grammar, start='value')

text = "max(mat1/mat2) / lag(mat1, 5)"
tree = my_parser.parse(text)
print(*MyTransformer().transform(tree), sep='\n')

