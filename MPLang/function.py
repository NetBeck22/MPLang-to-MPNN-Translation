from MPLang.mplangExpression import MPLangExpression, MPLangType, MPNN, MPNNLayer
from Graph.graph import Graph
from Function.activationFunction import activationFunction
import numpy as np

# Function MPLang expression e=f(e1)
#-----------------------------
# Stores the subexpression and function.
# Can create the MPNN and evaluate the expressions.

class Function(MPLangExpression):
    def __init__(self, func:activationFunction, expr):
        super().__init__(MPLangType.FUNC)
        self.__function = func
        self.__expression = expr
    
    def evaluate(self, feat:np.ndarray, graph:Graph):
        output = self.__expression.evaluate(feat, graph)
        a = np.empty(len(feat))
        for i in range(len(output)):
            a[i] =  self.__function.do(output[i])
        return a

    def createMPNN(self, d:int) -> MPNN:
        mpnn = self.__expression.createMPNN(d)
        super().addMPNN(mpnn)
        mpnn.addLayer(MPNNLayer(np.array([1]), 
                                np.array([0]), 
                                np.array([0]), 
                                self.__function))
        return self._mpnn
          