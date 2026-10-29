from MPLang.mplangExpression import MPLangExpression, MPLangType, MPNN, MPNNLayer
from Graph.graph import Graph
import numpy as np
from Function.activationFunction import activationFunction

# Sum MPLang expression e = e1 + e2
#-----------------------------
# Stores the subexpressions.
# Can create the MPNN and evaluate the expressions.

class Sum(MPLangExpression):
    def __init__(self, exp1:MPLangExpression, exp2:MPLangExpression):
        super().__init__(MPLangType.SUM)
        self.__expression1 = exp1
        self.__expression2 = exp2
    
    def __str__(self):
        return f"{self.__expression1} + {self.__expression2}"
    
    def evaluate(self, feat:np.ndarray, graph:Graph):
        return self.__expression1.evaluate(feat, graph) + self.__expression2.evaluate(feat, graph)

    def createMPNN(self, d:int) -> MPNN:
        mpnn1 = self.__expression1.createMPNN(d)
        mpnn2 = self.__expression2.createMPNN(d)
        super().addMPNN(mpnn1.concatenation(mpnn2))

        super().getMPNN().addLayer(MPNNLayer(np.array([1,1]), 
                                np.array([0,0]), 
                                np.array([0]), 
                                activationFunction('id')))
        return self._mpnn