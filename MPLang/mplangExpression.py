from enum import Enum
from MPNN.mpnn import MPNN, MPNNLayer
from Graph.graph import Graph
from abc import ABC, abstractmethod
import numpy as np

# All the possible MPLang expressions
class MPLangType(Enum):
    NUM = 1
    COMP = 2
    PROD = 3
    FUNC = 4
    DIAM = 5
    SUM = 6

# MPLangExpression
#--------------------------------
# Is an abstract dataclass, for all the MPLang expressions to derive.
# It has the type of expression and the MPNN that was made for the expression.
# We can add and get the MPNN.
# The derived MPLang expressions, need to implement the methods: createMPNN and evaluate.


class MPLangExpression(object):
    def __init__(self, type:MPLangType):
        self.__type = type
        self._mpnn = None

    def __str__(self):
        return f"{self._mpnn}"
    
    @abstractmethod
    def evaluate(self, feat:np.ndarray, graph:Graph):
        raise NotImplementedError("Subclasses must implement method")
    
    @abstractmethod
    def createMPNN(self, d:int) -> MPNN:
        raise NotImplementedError("Subclasses must implement method")
    
    def addMPNN(self, mpnn:MPNN):
        self._mpnn = mpnn
    
    def getMPNN(self):
        return self._mpnn

