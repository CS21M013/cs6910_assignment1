import numpy as np
class Layer:
    def __init__(self, hidden_units: int, activation:str=None):
        self.hidden_units = hidden_units
        self.activation = activation
        self.W = None
        self.b = None
        
    def initialize_params(self, n_in, hidden_units,init_type):
        np.random.seed(2)
        if init_type=="Random":
            self.W = 0.01*np.random.randn(n_in, hidden_units)
            self.b = 0.01*np.random.randn(1,hidden_units)

        elif init_type=="Xavier":
            self.W = np.random.randn(n_in, hidden_units) * np.sqrt(2/n_in) 
            self.b = np.zeros((1, hidden_units))

    def activation_fn(self, z, derivative=False):
        if self.activation == 'relu':
            if derivative:
                return np.where(z<=0,0,1)
            return np.maximum(0, z)
        if self.activation == 'sigmoid':
            if derivative:
                return (1 / (1 + np.exp(-z))) * (1-(1 / (1 + np.exp(-z))))
            return (1 / (1 + np.exp(-z)))
        if self.activation == 'tanh':
            t=(np.exp(z)-np.exp(-z))/(np.exp(z)+np.exp(-z))
            if derivative:
                return (1-t**2)
            return t

        if self.activation == 'softmax':
            if derivative: 
                exp = np.exp(z - np.max(z, axis=1, keepdims=True))
                return exp / np.sum(exp, axis=0) * (1 - exp / np.sum(exp, axis=0))
            exp = np.exp(z - np.max(z, axis=1, keepdims=True))
            return exp / np.sum(exp, axis=1, keepdims=True)

    def __repr__(self):
        return str(f'''Hidden Units={self.hidden_units}; Activation={self.activation}''')