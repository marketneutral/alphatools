from __future__ import print_function
import itertools
from lark import Lark, Transformer

grammar = r"""

    ?value: "(" value ")"
          | mylocalvar
          | add
          | SIGNED_NUMBER       -> number

    mylocalvar: "mylocalvar"
    add: value "+" value
    number: SIGNED_NUMBER

    %import common.SIGNED_NUMBER
    %import common.WS
    %ignore WS

    """

stack = []

class MyTransformer(Transformer):
    vcounter = itertools.count()

    def __init__(self):
        stack = []
        self.cmdlist = []

    def number(self, items):
        stack.append(items[0].value)
        
    def mylocalvar(self, items):
        stack.append('mylocalvar')

    def add(self, items):
        term2 = stack.pop()
        term1 = stack.pop()
        thisv = self.vcounter.next()
        stack.append('v' + str(thisv))
        self.cmdlist.append(
            'v' + str(thisv) + ' = ' + term1 + ' + ' + term2
        )
        
    def transform(self, tree):
        self._transform_tree(tree)
        v1 = stack.pop()
        self.cmdlist.append(
            'out[:] = ' + v1 + '[-1]'
        )

        return self.cmdlist


my_parser = Lark(grammar, start='value')


text = "mylocalvar + mylocalvar"
text = "mylocalvar + 2.5"
text = "2 + 2"
tree = my_parser.parse(text)
npcmds = MyTransformer().transform(tree)


