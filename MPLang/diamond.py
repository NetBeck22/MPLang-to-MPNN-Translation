from MPLang.mplangExpression import MPLangExpression, MPLangType, MPNN, MPNNLayer
from Graph.graph import Graph
import numpy as np
from Function.activationFunction import activationFunction

# Diamond MPLang expression e=diamond(e1)
#-----------------------------
# Stores the subexpression.
# Can create the MPNN and evaluate the expression.

class Diamond(MPLangExpression):
    def __init__(self, exp):
        super().__init__(MPLangType.DIAM)
        self.__expression = exp
    
    def evaluate(self, feat:np.ndarray, graph:Graph):
        a = np.empty(len(feat))
        output = self.__expression.evaluate(feat, graph)
        for i in range(len(feat)):
            som = 0
            for j in range(len(feat)):
                if graph.isEdge(i,j) :
                    som += output[j]
            a[i] = som
        return a

    def createMPNN(self, d:int) -> MPNN:
        mpnn = self.__expression.createMPNN(d)
        super().addMPNN(mpnn)
        mpnn.addLayer(MPNNLayer(np.array([0]), 
                                np.array([1]), 
                                np.array([0]), 
                                activationFunction('id')))
        return self._mpnn
          