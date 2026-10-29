import numpy as np

from MPLang.component import Component
from MPLang.function import Function
from MPLang.diamond import Diamond
from MPLang.number import Number
from MPLang.scalairProduct import ScalairProduct
from MPLang.sum import Sum

from Graph.graph import Graph
from Function.activationFunction import activationFunction

# create test MPLang expressions
exp1 = Number(5)
exp2 = Component(2)
exp3 = ScalairProduct(exp2, 5)
exp4 = Sum(exp1, exp2)
exp5 = Function(activationFunction("id"), exp2)
exp6 = Diamond(exp4)
exp7 = Sum(exp1, exp4)
exp8 = Diamond(ScalairProduct(exp7, 2))

expressions = [(exp1,"\nexpression1 = 5"), (exp2,"\nexpression2 = P2"),(exp3,"\nexpression3 = 5(exp2) = 5(P2)"),
               (exp4, "\nexpression4 = exp1 + exp2 = 5 + P2"),(exp5, "\nexpression5 = f(exp2) = f(P2), met f(x)=x"),
               (exp6, "\nexpression6 = diamond(exp2) = diamond(P2)"), (exp7, "\nexpression7 = exp1 + exp4 = 5 + (5 + P2)"), 
               (exp8, "\nexpression8 = diamond(2(exp7)) = diamond(2(5 + (5 + P2)))")]

# create graph
g = Graph([[0,1,1,0],[1,0,1,1],[1,1,0,1], [0,1,1,0]], np.array([[12,2,7,3], [10,1,15,2], [3,8,1,1], [25,4,1,5]]))
feat = g.getFeatureMap()
d = g.getNumberOfFeatures()

# print graph, expressions, output MPNN's and eval of MPLang and MPNN
def test():
    print(g)

    for i in expressions:
        MPNN = i[0].createMPNN(d) 
        MPLangEval = i[0].evaluate(feat,g)
        MPNNEval = MPNN.evaluate(feat,g)

        print(i[1])
        print(MPNN)
        print("MPNN: ", MPNNEval)
        print("MPLang: ", MPLangEval)

if __name__ == "__main__":
    test()

