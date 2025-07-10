import numpy as np
from activations import relu, sigmoid, linear

class Dense:
    def __init__(self, input_dim, units, activation='linear'):
        self.input_dim = input_dim
        self.units = units
        
        # Weight initialization (Xavier initialization)

        limit = np.sqrt(6 / (input_dim + units))
        self.weights = np.random.uniform(-limit, limit, (input_dim, units))
        self.bias = np.zeros(units)
        
        # Select activation function and derivative

        if activation == 'relu':
            self.activation = relu
            from activations.relu import relu_derivative
            self.activation_derivative = relu_derivative
        elif activation == 'sigmoid':
            self.activation = sigmoid
            from activations.sigmoid import sigmoid_derivative
            self.activation_derivative = sigmoid_derivative
        elif activation == 'linear':
            self.activation = linear
            from activations.linear import linear_derivative
            self.activation_derivative = linear_derivative
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Placeholders for forward/backward

        self.last_input = None
        self.last_z = None
        self.last_activation = None
        
        # Gradients

        self.dweights = None
        self.dbias = None
    
    def forward(self, inputs):
        self.last_input = inputs  # shape: (batch_size, input_dim)
        self.last_z = np.dot(inputs, self.weights) + self.bias  # pre-activation
        self.last_activation = self.activation(self.last_z)
        return self.last_activation
    
    def backward(self, dA):

        # dA is gradient of loss w.r.t. output of this layer (after activation)

        dZ = dA * self.activation_derivative(self.last_z)
        
        batch_size = self.last_input.shape[0]
        
        # Gradients

        self.dweights = np.dot(self.last_input.T, dZ) / batch_size
        self.dbias = np.sum(dZ, axis=0) / batch_size
        
        # Gradient w.r.t. input to pass to previous layer
        
        dinputs = np.dot(dZ, self.weights.T)
        return dinputs
    
    def update_params(self, learning_rate):
        self.weights -= learning_rate * self.dweights
        self.bias -= learning_rate * self.dbias