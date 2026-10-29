import numpy as np
from Function.activationFunction import activationFunction
from Graph.graph import Graph

# MPNN layer
#------------------------------------
# Has the 4 elements of the tuple that defines a MPNN layer: w
# W1, W2, bias, sigma-function
# All the elements have a get function and you can evaluate one layer.
 
class MPNNLayer(object):
    __w1 : np.ndarray
    __w2 : np.ndarray
    __bias : np.ndarray
    __sigma : activationFunction
    
    def __init__(self, w1:np.ndarray, w2:np.ndarray, bias:np.ndarray, sigma:activationFunction):
        self.__w1 = w1
        self.__w2 = w2
        self.__bias = bias.transpose()
        self.__sigma = sigma
    
    def __str__(self):
        return f"w1:{self.__w1},\nw2:{self.__w2},\nbias:{self.__bias},\nsigma:{self.__sigma.getType()}\n"
    
    def getStr(self):
        return self.__str__()
    
    def getw1(self) -> np.ndarray:
        return self.__w1
    
    def getw2(self) -> np.ndarray:
        return self.__w2
    
    def getBias(self)-> np.ndarray:
        return self.__bias
    
    def getSigma(self)-> activationFunction:
        return self.__sigma
    
    def evaluate(self, feat:np.ndarray,g:Graph):
        l = []

        for i in range(len(feat)):
            l.append(self.__sigma.do(np.add(np.add(np.matmul(self.__w1,feat[i])
                                ,np.matmul(self.__w2,self.__sum(feat, i,g))),
                                self.__bias)))
        return np.array(l)
    
    def __sum(self, feat:np.ndarray, i,g:Graph):
        sum = np.zeros(len(feat[0]))
        for j in range(len(feat)):
            if g.isEdge(i,j):
                sum +=  feat[j]
        return sum.transpose()
    
