from MPLang.mplangExpression import MPLangExpression, MPLangType, MPNN, MPNNLayer
from Graph.graph import Graph
import numpy as np
from Function.activationFunction import activationFunction

# Scalair Product MPLang expression e = ae1
#-----------------------------
# Stores the subexpression and n the number to scale.
# Can create the MPNN and evaluate the expression.

class ScalairProduct(MPLangExpression):
    def __init__(self, exp:MPLangExpression, num):
        super().__init__(MPLangType.PROD)
        self.__expression = exp
        self.__number = num
    
    def __str__(self):
        return f"{self.__expression} * {self.__number}"

    def evaluate(self, feat:np.ndarray, graph:Graph):
        return self.__expression.evaluate(feat, graph)*self.__number
    
    def createMPNN(self, d:int) -> MPNN:
        mpnn = self.__expression.createMPNN(d)
        super().addMPNN(mpnn)
        mpnn.addLayer(MPNNLayer(np.array([self.__number]), 
                                np.array([0]), 
                                np.array([0]), 
                                activationFunction('id')))
        return self._mpnn
          