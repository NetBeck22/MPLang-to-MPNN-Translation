from MPLang.mplangExpression import MPLangExpression, MPLangType, MPNN, MPNNLayer
from Graph.graph import Graph
import numpy as np
from Function.activationFunction import activationFunction

# Number MPLang expression e = n
#-----------------------------
# Stores the constante number n.
# Can create the MPNN and evaluate the expression.

class Number(MPLangExpression):
    def __init__(self, num):
        super().__init__(MPLangType.NUM)
        self.__number = num

    def __str__(self):
        return f"{self.__number}"
    
    def evaluate(self, feat:np.ndarray, graph:Graph):
        return np.full(len(feat), self.__number)
    
    def createMPNN(self, d:int) -> MPNN:
        super().addMPNN(MPNN([MPNNLayer(np.zeros(d), 
                                       np.zeros(d), 
                                       np.array([self.__number]), 
                                       activationFunction('id'))]))
        return self._mpnn