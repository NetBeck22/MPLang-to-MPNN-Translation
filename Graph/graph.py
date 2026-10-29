import numpy as np
from dataclasses import dataclass

# Graph
#----------------------
# Has the adjacencymatrix, featuremap, f = number of features, list of edges and list of nodes
# It's a data class. We can get information of the graph, but not modify it.
 
@dataclass
class Graph:
    # Symmetric adjacency matrix
    # Features in rowvectors, order of the adjacencymatrix
    def __init__(self, adj:list[list[int]], feat:list[list[int]]):
        self.__adjacencyMatrix = adj
        self.__featureMap = feat
        self.__f = len(self.__featureMap[0])
        self.__edges = []
        self.__nodes = []
        self.__createNodes()
        self.__createEdges()

    def __str__(self) -> str:
        feat = ""
        for i in self.__nodes:
            feat += f"\nNode {i} : {self.__featureMap[i]}"
        return f"Graph\nNodes: {self.__nodes}\nEdges: {self.__edges}\nFeatures {feat}" 
    
    def __createNodes(self):
        for i in range(len(self.__adjacencyMatrix)):
            self.__nodes.append(i)
    

    def __createEdges(self):
        for i in range(len(self.__adjacencyMatrix)):
            for j in range(i):
                if self.__adjacencyMatrix[i][j] != 0:
                    self.__edges.append((i,j))

    def getFeatureMap(self):
        return self.__featureMap
    
    def getNumberOfFeatures(self):
        return self.__f
    
    def isEdge(self,i,j):
        return self.__adjacencyMatrix[i][j] == 1



    
