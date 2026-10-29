from MPNN.mpnnLayer import MPNNLayer, activationFunction
from Graph.graph import Graph
import numpy as np

# MPNN class
# ------------------------
# Has a list of MPNN-layers, you can get it with index or the whole list.
# Also has the MPNNOperator, that can modify the structure of the MPNN.
# The object always make sures that it stays a ReLu-MPNN, when adding a layer.
# You can also remove the last layer and concatenate two MPNN objects.


class MPNN(object):
    def __init__(self, layers:list[MPNNLayer]):
        self.__layers = layers
        self.__operator = MPNNOperator()

    def __str__(self) -> str:
        str = ""
        for i in range(len(self.__layers)):
            str += f"\nLayer {i+1}\n"
            str += self.__layers[i].getStr() 
        return str 

    def evaluate(self, feat:np.ndarray,g:Graph) -> np.ndarray:
        for i in range(len(self.__layers)):
            if i == 0:
                output = self.__layers[i].evaluate(feat,g)
            else:
                input = output
                output = self.__layers[i].evaluate(input,g)
            
        return np.concatenate(output.transpose(), axis=None)
    
    def getLayers(self) -> np.ndarray:
        return self.__layers
    
    def getLayerByIndex(self, i) -> MPNNLayer:
        return self.__layers[i]

    def removeLastLayer(self):
        self.__layers.remove(self.__layers[-1])
    
    def addLayer(self, layer:MPNNLayer):
        if(len(self.__layers) != 0 and self.__layers[-1].getSigma().getType()  != 'relu'):
            self.__operator.createReLu(layer, self)
        else:
            self.__layers.append(layer)
    
    def concatenation(self, mpnn):
        return self.__operator.concatenation(self, mpnn)
    
# MPNNOperator
#-------------------------------------------
# Can make a ReLu-MPNN, when adding a layer.
# And also can concatenate two MPNN's.
#--------------------------------------------
# Other private methods to make it possible are:
# concatenation, parallell composition of two layers 
# and creating extra layers for concatenation of two MPNN's.

class MPNNOperator(object):
    def createReLu(self, layer:MPNNLayer, mpnn:MPNN):
        lastLayer = mpnn.getLayerByIndex(-1)
        mpnn.removeLastLayer()
        mpnn.addLayer(self.__concatenationLayers(MPNNLayer(lastLayer.getw1(), 
                                                        lastLayer.getw2(), 
                                                        lastLayer.getBias(), 
                                                        activationFunction('relu')), 
                                                MPNNLayer(-1*lastLayer.getw1(), 
                                                        -1*lastLayer.getw2(), 
                                                        -1*lastLayer.getBias(), 
                                                        activationFunction('relu'))))
        
        mpnn.addLayer(MPNNLayer(np.concatenate((layer.getw1(), -layer.getw1()), axis=None), 
                                np.concatenate((layer.getw2(), -layer.getw2()), axis=None),
                                layer.getBias(), 
                                layer.getSigma()))

    def concatenation(self, mpnn1:MPNN, mpnn2:MPNN) -> MPNN:
        new_mpnn = MPNN([])
        diffNumberOfLayers = len(mpnn1.getLayers())-len(mpnn2.getLayers())

        if(diffNumberOfLayers > 0):
            self.__createLayers(mpnn2, diffNumberOfLayers)
        elif(diffNumberOfLayers < 0): 
            self.__createLayers(mpnn1, diffNumberOfLayers*(-1))

        for i in range(len(mpnn1.getLayers())):
            if(i == 0):
                new_mpnn.addLayer(self.__concatenationLayers(mpnn1.getLayerByIndex(i), mpnn2.getLayerByIndex(i)))
            else:
                new_mpnn.addLayer(self.__parallelComposition(mpnn1.getLayerByIndex(i), mpnn2.getLayerByIndex(i)))

        return new_mpnn

    def __concatenationLayers(self, layerOne:MPNNLayer, layerTwo:MPNNLayer) -> MPNNLayer:
        return MPNNLayer(np.vstack((layerOne.getw1(), layerTwo.getw1())),
                         np.vstack((layerOne.getw2(), layerTwo.getw2())),
                         np.concatenate((layerOne.getBias(), layerTwo.getBias()), axis=None),
                         layerOne.getSigma())
    
    def __parallelComposition(self, layerOne:MPNNLayer, layerTwo:MPNNLayer) -> MPNNLayer:

        w11 = layerOne.getw1() 
        w12 = layerTwo.getw1() 
        w21 = layerOne.getw2() 
        w22 = layerTwo.getw2() 
       
        if(len(w11.shape) == 2 and w11.shape[0] == 1):
            w11 = w11.reshape(w11.shape[1])
        if(len(w12.shape) == 2 and w12.shape[0] == 1):
            w12 = w12.reshape(w12.shape[1])
        if(len(w21.shape) == 2 and w21.shape[0] == 1):
            w21 = w21.reshape(w21.shape[1])
        if(len(w22.shape) == 2 and w22.shape[0] == 1):
            w22 = w22.reshape(w22.shape[1])

    
        return MPNNLayer(np.vstack((np.hstack((w11,np.zeros(self.__createShape(w11,w12)))),
                                    np.hstack((np.zeros(self.__createShape(w12,w11)),w12)))),
                        np.vstack((np.hstack((w21,np.zeros(self.__createShape(w21,w22)))),
                                   np.hstack((np.zeros(self.__createShape(w22,w21)),w22)))),
                         np.hstack((layerOne.getBias(), layerTwo.getBias())),
                         layerOne.getSigma())
    
    def __createShape(self,row,column):
        shape = (1 if len(row.shape)==1 else row.shape[0],
                 column.shape[0] if len(column.shape) == 1 else column.shape[1])
        return shape if shape[0] != 1 else shape[1]

    def __createLayers(self, mpnn:MPNN, x:int):
        lastLayer = mpnn.getLayerByIndex(-1)
        shape = (len(lastLayer.getBias()), len(lastLayer.getBias()))

        if(len(mpnn.getLayers()) >= 2):
            mpnn.removeLastLayer()
            for i in range(x):
                mpnn.addLayer(MPNNLayer(np.full(shape, 1),
                                        np.zeros(shape),
                                        np.zeros(len(lastLayer.getBias())),
                                        activationFunction('relu'))) 
            mpnn.addLayer(lastLayer)
        else:
            mpnn.addLayer(MPNNLayer(np.full(shape,1), 
                                    np.zeros(shape), 
                                    np.zeros(len(lastLayer.getBias())), 
                                    activationFunction('id')))
            if (x > 1):
                self.__createLayers(mpnn, x-1)