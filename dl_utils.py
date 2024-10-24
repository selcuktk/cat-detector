import numpy as np

class activation:
    @staticmethod
    def sigmoid(z):
        return 1/(1 + np.exp(-z))
    
    @staticmethod
    def sigmoid_derivative(aL):
        return aL * (1 - aL)

    @staticmethod
    def leaky_relu(z, alpha=0.01):
        return np.where(z > 0, z, alpha * z)
    
    @staticmethod
    def leaky_relu_derivative(aL, alpha=0.01):
        daL = np.ones_like(aL)    # create an array which has the same size of dx
        daL[aL < 0] = alpha       # it changes the specific elements that provide the given condition.
        return daL
    
    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(aL):
        daL = np.ones_like(aL)  # create an array which has the same size of aL
        daL[aL <= 0] = 0       # it changes the specific elements that provide the given condition
        return daL

    

