from MPLang.mplangExpression import MPLangExpression, MPLangType, MPNN, MPNNLayer
from Graph.graph import Graph
import numpy as np
from Function.activationFunction import activationFunction

# Component MPLang expression e=Pn
#-----------------------------
# Stores n, of the n-th component.
# Can create the MPNN and evaluate the expression.

class Component(MPLangExpression):
    def __init__(self, n):
        super().__init__(MPLangType.COMP)
        self.__n = n
    
    def __str__(self):
        return f"P_{self.__n}"
    
    def evaluate(self, feat:np.ndarray, graph:Graph):
        a = np.empty(len(feat))
        for i in range(len(feat)):
            a[i] = feat[i][self.__n-1]
        return a

    def createMPNN(self, d:int) -> MPNN:
        w1 = np.zeros(d)
        w1[self.__n-1] = 1
        super().addMPNN(MPNN([MPNNLayer(w1, 
                                       np.zeros(d), 
                                       np.array([0]), 
                                       activationFunction('id'))]))
        return self._mpnn