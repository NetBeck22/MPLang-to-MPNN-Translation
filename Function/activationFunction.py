import numpy as np

# activationFunction
#-----------------------
# Function that you specify with a type and can execute.

class activationFunction:
    def __init__(self, type):
        self.__type = type
    
    def __relu(self, x):
        if(len(x)>1):
            a = np.empty(len(x))
            for i in range(len(x)):
                a[i] = max(0, x[i])
            return a

        else:
            return max(0,x)

    def __id(self, x):
        return x
    
    def do(self, x):
        if self.__type == "relu":
            return self.__relu(x)
        elif self.__type == "id":
            return self.__id(x)

    def getType(self):
        return self.__type